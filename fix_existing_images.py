#!/usr/bin/env python3
"""
Migrate previously-downloaded CivitAI media to use correct extensions.

The earlier version of civitai_to_obsidian.py saved every download as
`{image_id}.jpeg` regardless of the actual file format. This script
walks the configured Obsidian media folder, sniffs each file's magic
bytes, and:

  - renames image files whose extension doesn't match their bytes
  - deletes video files (MP4/etc.) that were saved with image extensions
  - rewrites the matching `![[...]]` references in vault markdown files
  - verifies every embed in scope still resolves to a file on disk

Scope is restricted to files produced by this project: media inside
`obsidian.media_folder` and markdown inside `obsidian.base_directory`.
Pass --scan-all-markdown to widen the markdown scan to the whole vault.

Defaults to a dry run. Pass --apply to actually change files.

Usage:
    python fix_existing_images.py                   # dry run
    python fix_existing_images.py --apply           # do it
    python fix_existing_images.py --config foo.yaml # alt config
"""

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from civitai_to_obsidian import load_config

# Magic byte signatures we want to recognize, including formats we
# explicitly do NOT keep (videos). Returning the format name lets us
# distinguish "rename me" from "delete me".
_SIGNATURES: List[Tuple[bytes, str]] = [
    (b'\xff\xd8\xff', 'jpeg'),
    (b'\x89PNG\r\n\x1a\n', 'png'),
    (b'GIF87a', 'gif'),
    (b'GIF89a', 'gif'),
]

# Extensions that count as "image" in this project. Anything else
# (mp4, mov, etc.) is treated as a video to be removed.
_IMAGE_EXTS = {'jpeg', 'jpg', 'png', 'webp', 'gif'}

# Captures the inside of an Obsidian embed: `![[ ... ]]`. The inside
# may contain `|alias` and/or `#section` qualifiers, which we preserve.
_EMBED_RE = re.compile(r'!\[\[([^\]]+?)\]\]')


def detect_format(head: bytes) -> Optional[str]:
    """Return canonical format name (jpeg/png/webp/gif/mp4) or None."""
    for sig, name in _SIGNATURES:
        if head.startswith(sig):
            return name
    if len(head) >= 12 and head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'webp'
    # ISO Base Media (MP4, MOV, etc.) — `ftyp` box at offset 4.
    if len(head) >= 12 and head[4:8] == b'ftyp':
        return 'mp4'
    return None


def scan_media_folder(
    media_root: Path
) -> Tuple[List[Tuple[Path, str, str]], List[Path], List[Path]]:
    """Walk media_root and classify every file.

    Returns three lists:
      renames: [(path, current_ext, correct_ext), ...]
      videos:  paths to delete
      broken:  paths we couldn't classify (zero-byte, unknown magic)
    """
    renames: List[Tuple[Path, str, str]] = []
    videos: List[Path] = []
    broken: List[Path] = []

    for path in media_root.rglob('*'):
        if not path.is_file():
            continue

        try:
            with open(path, 'rb') as f:
                head = f.read(16)
        except OSError as e:
            print(f"  ⚠️  Could not read {path}: {e}")
            broken.append(path)
            continue

        if not head:
            broken.append(path)
            continue

        fmt = detect_format(head)
        if fmt is None:
            broken.append(path)
            continue

        if fmt not in _IMAGE_EXTS:
            videos.append(path)
            continue

        current_ext = path.suffix.lower().lstrip('.')
        # Treat .jpg and .jpeg as interchangeable for "already correct".
        canonical_current = 'jpeg' if current_ext == 'jpg' else current_ext
        if canonical_current != fmt:
            renames.append((path, current_ext, fmt))

    return renames, videos, broken


def iter_markdown_files(roots: List[Path]):
    """Yield every .md file under any of `roots`, deduplicated.

    Skips any path that lives under a hidden (dot-prefixed) directory,
    which excludes `.obsidian/`, our own `.civitai-*-backup-*/` trees,
    and similar machine-managed folders from the scan.
    """
    seen: Set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for md_path in root.rglob('*.md'):
            if any(part.startswith('.') for part in md_path.parts):
                continue
            resolved = md_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield md_path


def split_embed_ref(inner: str) -> Tuple[str, str]:
    """Split the inside of an `![[…]]` embed into (path, suffix).

    Suffix preserves `|alias`, `#section`, or `#^block` qualifiers so
    we can put them back when rewriting. Whitespace around the path
    is stripped — Obsidian tolerates it on input but it's not part of
    the identity of the reference.
    """
    # Find the first qualifier character; everything before is the path.
    for i, ch in enumerate(inner):
        if ch in ('|', '#'):
            return inner[:i].strip(), inner[i:]
    return inner.strip(), ''


def backup_files(
    paths: Iterable[Path],
    vault_path: Path,
    backup_root: Path
) -> int:
    """Copy each path into `backup_root` preserving its vault-relative
    layout. Returns the count of files actually copied.

    Files already present in the backup root are not re-copied (this
    matters when the user re-runs the script after a partial failure).
    """
    backup_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in paths:
        try:
            rel = path.relative_to(vault_path)
        except ValueError:
            # Outside the vault — store under _outside/ keyed by abs path.
            rel = Path('_outside') / Path(*path.parts[1:])
        target = backup_root / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # copy2 preserves mtime / mode — restore is then a plain `cp -r`.
        shutil.copy2(path, target)
        copied += 1
    return copied


def write_atomic(path: Path, content: str) -> None:
    """Write text atomically: tempfile in same dir then os.replace.

    POSIX `rename` is atomic on the same filesystem, so a crash mid-
    write either leaves the original intact or replaces it with the
    fully-written new content — never a half-written file.
    """
    tmp = path.with_name(path.name + '.tmp.civitai_fix')
    try:
        with open(tmp, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def check_rewrite_invariants(old: str, new: str) -> Optional[str]:
    """Return None if the rewrite is structurally safe, else a reason.

    These invariants must hold because we only do text replacement
    inside `![[...]]` brackets — nothing outside that pattern should
    change. If one fails the rewrite is suspect and we'd rather skip.
    """
    if old.count('\n') != new.count('\n'):
        return (
            f"line count changed ({old.count(chr(10))} → "
            f"{new.count(chr(10))})"
        )
    if old.count('![[') != new.count('![['):
        return (
            f"![[ count changed ({old.count('![[')} → "
            f"{new.count('![[')})"
        )
    if old.count(']]') != new.count(']]'):
        return (
            f"]] count changed ({old.count(']]')} → "
            f"{new.count(']]')})"
        )
    if not new:
        return "result is empty"
    return None


def rewrite_markdown(
    md_paths,
    rename_map: Dict[str, str],
    dry_run: bool
) -> Tuple[int, int, int]:
    """Rewrite embed paths in markdown files using rename_map.

    Preserves `|alias` and `#section` qualifiers, ignores embeds whose
    path isn't in rename_map, and rewrites multiple embeds per line
    correctly. Returns (refs_updated, files_updated, files_skipped).
    Files are skipped (not written) if a safety invariant fails.
    """
    refs_updated = 0
    files_updated = 0
    files_skipped = 0

    for md_path in md_paths:
        try:
            text = md_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue

        file_updates = 0

        def replace(match: 're.Match[str]') -> str:
            nonlocal file_updates
            inner = match.group(1)
            path, suffix = split_embed_ref(inner)
            new_path = rename_map.get(path)
            if new_path is None:
                return match.group(0)
            file_updates += 1
            return f'![[{new_path}{suffix}]]'

        new_text = _EMBED_RE.sub(replace, text)

        if not file_updates:
            continue

        problem = check_rewrite_invariants(text, new_text)
        if problem:
            print(f"  ⚠️  Skipping {md_path}: invariant failed — {problem}")
            files_skipped += 1
            continue

        refs_updated += file_updates
        files_updated += 1
        if not dry_run:
            write_atomic(md_path, new_text)

    return refs_updated, files_updated, files_skipped


def count_external_media_refs(
    vault_path: Path,
    media_root: Path,
    in_scope_md_files
) -> Tuple[int, int]:
    """Count media embeds in markdown OUTSIDE the in-scope file set.

    Surfaces references the targeted rewrite won't touch — useful when
    the user has manually embedded these images in notes outside
    `base_directory`. Returns (ref_count, file_count).
    """
    in_scope = {p.resolve() for p in in_scope_md_files}
    media_prefix = media_root.relative_to(vault_path).as_posix() + '/'
    ref_count = 0
    file_count = 0

    for md_path in vault_path.rglob('*.md'):
        if any(part.startswith('.') for part in md_path.parts):
            continue
        if md_path.resolve() in in_scope:
            continue
        try:
            text = md_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        file_hits = 0
        for match in _EMBED_RE.finditer(text):
            path, _ = split_embed_ref(match.group(1))
            if path.startswith(media_prefix):
                file_hits += 1
        if file_hits:
            ref_count += file_hits
            file_count += 1
    return ref_count, file_count


def verify_embeds(
    md_paths,
    vault_path: Path,
    media_root: Path,
    rename_map: Dict[str, str],
    deletions: Set[Path]
) -> List[Tuple[Path, int, str]]:
    """Project the post-apply file set and verify every in-scope embed.

    We don't read disk directly because in dry-run mode disk still
    reflects the pre-migration state. Instead we compute what will
    exist *after* renames and deletions, project every embed through
    rename_map (mirroring what `rewrite_markdown` does), and check
    membership. In apply mode the projection matches reality, so the
    same code serves both.

    Returns (markdown_file, line_number, embed_ref_as_seen_pre_apply).
    """
    projected: Set[str] = set()
    for path in media_root.rglob('*'):
        if not path.is_file() or path in deletions:
            continue
        try:
            rel = path.relative_to(vault_path).as_posix()
        except ValueError:
            continue
        projected.add(rename_map.get(rel, rel))

    media_prefix = media_root.relative_to(vault_path).as_posix() + '/'
    broken: List[Tuple[Path, int, str]] = []

    for md_path in md_paths:
        try:
            text = md_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue

        for line_no, line in enumerate(text.splitlines(), 1):
            for match in _EMBED_RE.finditer(line):
                path, _ = split_embed_ref(match.group(1))
                if not path.startswith(media_prefix):
                    continue
                final_path = rename_map.get(path, path)
                if final_path not in projected:
                    broken.append((md_path, line_no, path))
    return broken


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fix file extensions on previously-downloaded CivitAI media "
            "and update Obsidian markdown references to match."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform changes. Without this flag, runs as a "
             "dry run that only reports what would change."
    )
    parser.add_argument(
        "--delete-videos",
        action="store_true",
        help="Permanently delete video files. Default is to quarantine "
             "them under the backup directory so you can review first."
    )
    parser.add_argument(
        "--scan-all-markdown",
        action="store_true",
        help="Scan the entire vault for markdown references instead of "
             "just `obsidian.base_directory`. Useful if you've manually "
             "embedded these images in notes outside the managed tree."
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Where to store backups of touched markdown and quarantined "
             "videos. Defaults to "
             "`<vault>/.civitai-fix-backup-<timestamp>/`."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    obs = config.get("obsidian", {})
    vault_path = Path(obs.get("vault_path", ".")).resolve()
    media_rel = obs.get(
        "media_folder", "zzMedia/Model and Lora Example Images"
    )
    base_dir_rel = obs.get("base_directory", "")
    media_root = (vault_path / media_rel).resolve()
    base_dir = (vault_path / base_dir_rel).resolve() if base_dir_rel else None

    if not media_root.exists():
        print(f"❌ Media folder not found: {media_root}", file=sys.stderr)
        sys.exit(1)

    if args.scan_all_markdown:
        md_roots = [vault_path]
        scope_label = f"entire vault ({vault_path})"
    elif base_dir and base_dir.exists():
        md_roots = [base_dir]
        scope_label = f"base_directory ({base_dir})"
    else:
        print(
            "❌ obsidian.base_directory is not set or does not exist, "
            "and --scan-all-markdown was not passed. Refusing to guess.",
            file=sys.stderr
        )
        sys.exit(1)

    if args.backup_dir:
        backup_root = Path(args.backup_dir).resolve()
    else:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_root = vault_path / f'.civitai-fix-backup-{stamp}'

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"🔍 {mode}")
    print(f"   Media scope:    {media_root}")
    print(f"   Markdown scope: {scope_label}")
    print(f"   Backup dir:     {backup_root}\n")

    renames, videos, broken = scan_media_folder(media_root)

    print(f"  Files needing rename:    {len(renames)}")
    print(f"  Video files (to remove): {len(videos)}")
    print(f"  Broken / unknown files:  {len(broken)}\n")

    # Build the rename map keyed by vault-relative POSIX paths — that's
    # the format Obsidian embeds use, and it's stable across platforms.
    rename_map: Dict[str, str] = {}
    for path, _, new_ext in renames:
        new_path = path.with_suffix(f'.{new_ext}')
        try:
            old_rel = path.relative_to(vault_path).as_posix()
            new_rel = new_path.relative_to(vault_path).as_posix()
        except ValueError:
            continue
        rename_map[old_rel] = new_rel

    # Sample preview
    for path, cur, new in renames[:5]:
        print(f"  rename: {path.name} ({cur} → {new})")
    if len(renames) > 5:
        print(f"  ... and {len(renames) - 5} more renames")
    for path in videos[:5]:
        action = "delete" if args.delete_videos else "quarantine"
        print(f"  video : {path.name} ({action})")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more videos")
    for path in broken[:5]:
        print(f"  broken: {path.name} (skipping — manual review)")
    if len(broken) > 5:
        print(f"  ... and {len(broken) - 5} more broken")
    print()

    # Collect the markdown file list once so dry-run and apply behave
    # identically. Identify which files would be touched (so we know
    # which to back up).
    md_files = list(iter_markdown_files(md_roots))
    files_to_touch: List[Path] = []
    for md_path in md_files:
        try:
            text = md_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        for match in _EMBED_RE.finditer(text):
            path, _ = split_embed_ref(match.group(1))
            if path in rename_map:
                files_to_touch.append(md_path)
                break

    # APPLY PHASE — order matters:
    #   1. Backup every markdown we'll modify (so any failure is
    #      reversible by `cp -r`).
    #   2. Rewrite markdown atomically.
    #   3. Rename image files on disk (now matching the rewritten
    #      refs).
    #   4. Quarantine videos last — least reversible operation,
    #      done only after everything else succeeded.
    rename_failures = 0
    backup_count = 0
    if args.apply:
        backup_count = backup_files(
            files_to_touch, vault_path, backup_root
        )
        print(
            f"💾 Backed up {backup_count} markdown file(s) to "
            f"{backup_root}"
        )

    refs_updated, files_updated, files_skipped = rewrite_markdown(
        md_files, rename_map, dry_run=not args.apply
    )

    if args.apply:
        for path, _, new_ext in renames:
            target = path.with_suffix(f'.{new_ext}')
            if target.exists() and target != path:
                print(f"  ⚠️  target exists, skipping: {target}")
                rename_failures += 1
                continue
            try:
                path.rename(target)
            except OSError as e:
                print(f"  ⚠️  rename failed for {path}: {e}")
                rename_failures += 1

        # Quarantine (or delete) videos
        videos_handled = 0
        for path in videos:
            try:
                if args.delete_videos:
                    path.unlink()
                else:
                    try:
                        rel = path.relative_to(vault_path)
                    except ValueError:
                        rel = Path('_outside') / path.name
                    quarantine_target = backup_root / 'videos' / rel
                    quarantine_target.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    # If a same-name file already exists in quarantine
                    # (re-run), suffix with a counter to keep both.
                    if quarantine_target.exists():
                        stem = quarantine_target.stem
                        suf = quarantine_target.suffix
                        i = 1
                        while True:
                            candidate = quarantine_target.with_name(
                                f"{stem}.{i}{suf}"
                            )
                            if not candidate.exists():
                                quarantine_target = candidate
                                break
                            i += 1
                    shutil.move(str(path), str(quarantine_target))
                videos_handled += 1
            except OSError as e:
                print(f"  ⚠️  video op failed for {path}: {e}")

    # Surface any references in markdown OUTSIDE the targeted scope
    # so the user can decide whether to widen with --scan-all-markdown.
    external_refs, external_files = 0, 0
    if not args.scan_all_markdown:
        external_refs, external_files = count_external_media_refs(
            vault_path, media_root, md_files
        )

    print("📊 Summary")
    print(f"  Renamed files:           {len(renames) - rename_failures}")
    video_action = "deleted" if args.delete_videos else "quarantined"
    print(f"  Videos {video_action}:        {len(videos)}")
    print(f"  Markdown refs updated:   {refs_updated}")
    print(f"  Markdown files touched:  {files_updated}")
    if files_skipped:
        print(
            f"  Markdown files SKIPPED:  {files_skipped} "
            f"(invariant failure — see warnings above)"
        )
    print(f"  Broken files (skipped):  {len(broken)}")
    if args.apply:
        print(f"  Backups written to:      {backup_root}")

    if external_refs:
        print(
            f"\n⚠️  {external_refs} embed(s) in {external_files} markdown "
            f"file(s) outside `{base_dir_rel}` reference these images.\n"
            f"   The targeted run won't touch them, so any whose extension "
            f"changes will break after apply.\n"
            f"   Re-run with --scan-all-markdown to include the entire "
            f"vault."
        )

    # Verification — projects the post-apply state so dry-run and
    # apply both report the same truth: "what will display in Obsidian
    # after this migration runs?"
    print("\n🔎 Verifying every in-scope media embed will resolve "
          "after apply...")
    # Videos disappear from the media folder whether they're deleted
    # or quarantined — both look like deletions to the projected view.
    dangling = verify_embeds(
        md_files,
        vault_path,
        media_root,
        rename_map=rename_map,
        deletions=set(videos),
    )
    if not dangling:
        print("  ✅ Every in-scope media embed will resolve after apply.")
    else:
        by_file: Dict[Path, List[Tuple[int, str]]] = {}
        for md_path, line_no, ref in dangling:
            by_file.setdefault(md_path, []).append((line_no, ref))
        total = len(dangling)
        print(
            f"  ⚠️  {total} embed(s) in {len(by_file)} file(s) will be "
            f"broken after apply (mostly removed videos). "
            f"Listed by file:line — review and delete manually:"
        )
        for md_path, items in sorted(by_file.items()):
            try:
                pretty = md_path.relative_to(vault_path)
            except ValueError:
                pretty = md_path
            for line_no, ref in items:
                print(f"    {pretty}:{line_no}  →  {ref}")

    if not args.apply:
        print(
            "\n💡 This was a dry run. Re-run with --apply to commit "
            "these changes."
        )


if __name__ == "__main__":
    main()
