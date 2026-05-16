#!/usr/bin/env python3
"""
Rename existing image folders from `{model_id}_v{version_id}` style to
`{model_name} ({version_name})` so the vault is browsable by model
name. Updates every matching `![[…]]` reference in markdown to point
at the new folder name.

The CivitAI API is queried once per unique model_id (results cached
to a JSON file next to the script so re-runs don't re-fetch). Folder
collisions are detected up-front and reported before any change.

Reuses the safety pattern from fix_existing_images.py: dry run by
default, markdown backups, atomic writes, rewrite invariants.

Usage:
    python rename_model_folders.py                # dry run
    python rename_model_folders.py --apply        # do it
"""

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from civitai_to_obsidian import (
    CivitAIFetcher,
    ObsidianPageGenerator,
    load_config,
)
from fix_existing_images import (
    _EMBED_RE,
    backup_files,
    iter_markdown_files,
    rewrite_markdown,
    split_embed_ref,
)

# Folders the downloader has historically created look like one of:
#   {model_id}                 e.g. 443821
#   {model_id}_v{version_id}   e.g. 470073_v522995
# Older legacy folders use human-readable names (e.g. "0x00F Ashley-Wood")
# and are left alone — they're already in the desired style.
_FOLDER_RE = re.compile(r'^(\d+)(?:_v(\d+))?$')


def parse_folder(name: str) -> Optional[Tuple[int, Optional[int]]]:
    """Parse a folder name into (model_id, optional version_id)."""
    m = _FOLDER_RE.match(name)
    if not m:
        return None
    model_id = int(m.group(1))
    version_id = int(m.group(2)) if m.group(2) else None
    return model_id, version_id


class ModelCache:
    """Disk-backed cache of model_data keyed by model_id.

    Each entry stores the raw API response. We only need name and
    modelVersions, but storing the full payload lets future scripts
    re-use the same cache file. A `null` entry marks "API said 404 or
    similar" so we don't keep retrying.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Optional[Dict[str, Any]]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, model_id: int) -> Optional[Dict[str, Any]]:
        return self._data.get(str(model_id))

    def has(self, model_id: int) -> bool:
        return str(model_id) in self._data

    def set(
        self, model_id: int, data: Optional[Dict[str, Any]]
    ) -> None:
        self._data[str(model_id)] = data

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, indent=2), encoding='utf-8'
        )


def fetch_model(
    fetcher: CivitAIFetcher,
    model_id: int,
    cache: ModelCache,
    api_delay: float
) -> Optional[Dict[str, Any]]:
    """Fetch a model from the API, or return cached value."""
    if cache.has(model_id):
        return cache.get(model_id)
    try:
        data = fetcher.get_model_details(model_id)
        cache.set(model_id, data)
    except Exception as e:
        print(f"  ⚠️  API failed for model {model_id}: {e}")
        cache.set(model_id, None)
        data = None
    # Save after every fetch so a crash mid-run preserves progress.
    cache.save()
    time.sleep(api_delay)
    return data


def plan_renames(
    media_root: Path,
    fetcher: CivitAIFetcher,
    cache: ModelCache,
    api_delay: float
) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    """Walk media_root and plan folder renames.

    Returns:
      renames:  [(old_folder, new_folder), ...] — to be renamed
      skipped:  folders we left alone (legacy names or API failure)
      collisions: folders whose target name collides with another
                  source folder or an existing unrelated folder
    """
    renames: List[Tuple[Path, Path]] = []
    skipped: List[Path] = []
    collisions: List[Path] = []
    proposed_targets: Dict[Path, Path] = {}
    proposed_names: Set[str] = set()

    subfolders = sorted(
        p for p in media_root.iterdir() if p.is_dir()
    )

    for folder in subfolders:
        parsed = parse_folder(folder.name)
        if parsed is None:
            skipped.append(folder)
            continue
        model_id, version_id = parsed

        data = fetch_model(fetcher, model_id, cache, api_delay)
        if not data:
            print(
                f"  ⚠️  No model data for {folder.name} — skipping rename"
            )
            skipped.append(folder)
            continue

        new_name = ObsidianPageGenerator.build_image_folder_name(
            data, version_id
        )
        target = media_root / new_name

        # Collision rules:
        #  1. Same target proposed by two different source folders
        #     (e.g. version not in API anymore → both fall back to
        #      same fallback name).
        #  2. Target name matches an existing folder we are NOT
        #     planning to rename away.
        will_collide = False
        if new_name in proposed_names:
            collisions.append(folder)
            will_collide = True
        elif target.exists() and target != folder:
            # If the existing target is itself in our rename set (will
            # move away), that's fine. Otherwise it's a collision.
            target_will_be_renamed = any(
                old == target for old, _ in proposed_targets.items()
            )
            if not target_will_be_renamed:
                collisions.append(folder)
                will_collide = True

        if will_collide:
            continue

        if target == folder:
            # Already named correctly; nothing to do.
            skipped.append(folder)
            continue

        proposed_targets[folder] = target
        proposed_names.add(new_name)
        renames.append((folder, target))

    return renames, skipped, collisions


def build_file_rename_map(
    vault_path: Path,
    renames: List[Tuple[Path, Path]]
) -> Dict[str, str]:
    """Per-image-file rename map for markdown rewrite.

    For each renamed folder, enumerate the files inside and map their
    vault-relative POSIX path from old to new. This lets us reuse
    rewrite_markdown() from fix_existing_images.py unchanged.
    """
    rename_map: Dict[str, str] = {}
    for old_folder, new_folder in renames:
        for old_file in old_folder.rglob('*'):
            if not old_file.is_file():
                continue
            relative = old_file.relative_to(old_folder)
            new_file = new_folder / relative
            try:
                old_rel = old_file.relative_to(vault_path).as_posix()
                new_rel = new_file.relative_to(vault_path).as_posix()
            except ValueError:
                continue
            rename_map[old_rel] = new_rel
    return rename_map


def find_files_to_touch(
    md_files: List[Path],
    rename_map: Dict[str, str]
) -> List[Path]:
    """Identify markdown files containing any embed we will rewrite."""
    to_touch: List[Path] = []
    for md_path in md_files:
        try:
            text = md_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        for match in _EMBED_RE.finditer(text):
            path, _ = split_embed_ref(match.group(1))
            if path in rename_map:
                to_touch.append(md_path)
                break
    return to_touch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename image folders from `{model_id}_v{version_id}` to "
            "`{model_name} ({version_name})` and update markdown refs."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--cache-file",
        default=".model_cache.json",
        help="Where to store cached API responses (default: "
             ".model_cache.json next to this script)."
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Backup directory. Defaults to "
             "<vault>/.civitai-rename-backup-<timestamp>/"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    obs = config.get("obsidian", {})
    vault_path = Path(obs.get("vault_path", ".")).resolve()
    media_rel = obs.get(
        "media_folder", "zzMedia/Model and Lora Example Images"
    )
    media_root = (vault_path / media_rel).resolve()

    if not media_root.exists():
        print(f"❌ Media folder not found: {media_root}", file=sys.stderr)
        sys.exit(1)

    civ = config.get("civitai", {})
    rl = config.get("rate_limits", {})
    fetcher = CivitAIFetcher(
        api_key=civ.get("api_key"),
        base_url=civ.get("base_url", "https://civitai.com/api/v1"),
        max_retries=rl.get("max_retries", 3),
        backoff_factor=rl.get("backoff_factor", 1),
    )
    api_delay = rl.get("api_delay", 1.5)

    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        cache_path = Path(__file__).parent / cache_path
    cache = ModelCache(cache_path)

    if args.backup_dir:
        backup_root = Path(args.backup_dir).resolve()
    else:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_root = vault_path / f'.civitai-rename-backup-{stamp}'

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"🔍 {mode}")
    print(f"   Media root:    {media_root}")
    print(f"   Cache file:    {cache_path}")
    print(f"   Backup dir:    {backup_root}\n")

    print("🌐 Fetching model names from CivitAI API "
          "(cached on disk for re-runs)...")
    renames, skipped, collisions = plan_renames(
        media_root, fetcher, cache, api_delay
    )

    print(f"\n  Folders to rename:    {len(renames)}")
    print(f"  Folders skipped:      {len(skipped)} "
          f"(legacy name, API failure, or already correct)")
    print(f"  Folder name collisions:{len(collisions)}\n")

    if collisions:
        print("⚠️  These source folders would collide on the chosen "
              "target name. Resolve manually before re-running:")
        for path in collisions:
            print(f"    {path.name}")
        print()

    for old, new in renames[:10]:
        print(f"  rename: {old.name}  →  {new.name}")
    if len(renames) > 10:
        print(f"  ... and {len(renames) - 10} more")
    print()

    if not renames:
        print("✅ Nothing to rename.")
        return

    rename_map = build_file_rename_map(vault_path, renames)
    md_files = list(iter_markdown_files([vault_path]))
    files_to_touch = find_files_to_touch(md_files, rename_map)

    print(
        f"📝 Will rewrite refs in {len(files_to_touch)} markdown file(s) "
        f"({len(rename_map)} file-paths affected).\n"
    )

    if args.apply:
        backup_count = backup_files(
            files_to_touch, vault_path, backup_root
        )
        print(
            f"💾 Backed up {backup_count} markdown file(s) to "
            f"{backup_root}\n"
        )

    refs_updated, files_updated, files_skipped = rewrite_markdown(
        md_files, rename_map, dry_run=not args.apply
    )

    if args.apply:
        # Rename folders last so backups + markdown rewrites complete
        # first. A failure here leaves markdown pointing at the new
        # names while disk still has old names, which is recoverable
        # by re-running this script.
        rename_failures = 0
        for old, new in renames:
            try:
                shutil.move(str(old), str(new))
            except OSError as e:
                print(f"  ⚠️  rename failed {old} → {new}: {e}")
                rename_failures += 1
        renamed_count = len(renames) - rename_failures
    else:
        renamed_count = len(renames)

    print("📊 Summary")
    print(f"  Folders renamed:         {renamed_count}")
    print(f"  Markdown refs updated:   {refs_updated}")
    print(f"  Markdown files touched:  {files_updated}")
    if files_skipped:
        print(
            f"  Markdown files SKIPPED:  {files_skipped} "
            f"(invariant failure — see warnings above)"
        )
    if args.apply:
        print(f"  Backups written to:      {backup_root}")

    # Verify projected post-apply embed resolution.
    print("\n🔎 Verifying every media embed will resolve after apply...")
    media_prefix = media_root.relative_to(vault_path).as_posix() + '/'

    # Build projected set: every current file, with rename_map applied.
    projected: Set[str] = set()
    for path in media_root.rglob('*'):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(vault_path).as_posix()
        except ValueError:
            continue
        projected.add(rename_map.get(rel, rel))

    broken: List[Tuple[Path, int, str]] = []
    for md_path in md_files:
        try:
            text = md_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for match in _EMBED_RE.finditer(line):
                ref, _ = split_embed_ref(match.group(1))
                if not ref.startswith(media_prefix):
                    continue
                final = rename_map.get(ref, ref)
                if final not in projected:
                    broken.append((md_path, line_no, ref))

    if not broken:
        print("  ✅ Every media embed will resolve after apply.")
    else:
        by_file: Dict[Path, List[Tuple[int, str]]] = {}
        for md_path, line_no, ref in broken:
            by_file.setdefault(md_path, []).append((line_no, ref))
        print(
            f"  ⚠️  {len(broken)} embed(s) in {len(by_file)} file(s) "
            f"will be broken after apply. Most are likely pre-existing "
            f"breakage unrelated to this rename:"
        )
        for md_path, items in sorted(by_file.items())[:20]:
            try:
                pretty = md_path.relative_to(vault_path)
            except ValueError:
                pretty = md_path
            for line_no, ref in items[:3]:
                print(f"    {pretty}:{line_no}  →  {ref}")
        if len(by_file) > 20:
            print(f"    ... and {len(by_file) - 20} more files")

    if not args.apply:
        print(
            "\n💡 This was a dry run. Re-run with --apply to commit "
            "these changes."
        )


if __name__ == "__main__":
    main()
