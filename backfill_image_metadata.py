#!/usr/bin/env python3
"""
Backfill embedded generation metadata in previously-downloaded
CivitAI images.

For each image file under the configured Obsidian media folder, look
up its CivitAI image record by ID and merge any missing generation
parameters into the file's embedded metadata. Existing embedded data
is preserved unconditionally — only gaps are filled.

Mirrors fix_existing_images.py's safety conventions:
  - Dry-run by default; --apply is required to write
  - Optional --backup creates `<name>.civitai-orig` before first write
  - Per-file: tempfile + verify + atomic rename (see enrich_file)
  - PNGs: chunk-level surgery (pixel data never decoded)
  - JPEGs: piexif EXIF-only patch (compressed stream untouched)
  - WEBP/GIF: skipped (reported as unsupported)

Usage:
    # See what would change across every model folder
    python backfill_image_metadata.py

    # Apply to every model folder
    python backfill_image_metadata.py --apply

    # Scope to one model's folder (glob against folder name)
    python backfill_image_metadata.py --scope "Plant Milk*"

    # Apply with backups, slower API rate to be polite
    python backfill_image_metadata.py --apply --backup --api-delay 2.0
"""

import argparse
import fnmatch
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from civitai_to_obsidian import (
    CivitAIFetcher,
    EnrichStatus,
    enrich_file,
    load_config,
    parse_a1111_params,
)


# Filename pattern: `{numeric image id}.{ext}`. Anything else is
# skipped (e.g. files the user dropped in manually, or our own
# `.civitai-orig` backups).
_IMAGE_NAME_RE = re.compile(
    r"^(\d+)\.(png|jpg|jpeg|webp|gif)$", re.IGNORECASE
)

# A file is treated as "already complete" — and skipped when
# --skip-complete is set — if its embedded params include all of these
# keys. They're the core A1111 fields most downstream tools care
# about. Civitai-specific extras (resources, hashes, etc.) aren't on
# the list because they're not in API meta to begin with.
_COMPLETENESS_KEYS = frozenset({
    "prompt", "Steps", "Sampler", "CFG scale", "Seed", "Size", "Model",
})


def fetch_image_meta(
    fetcher: CivitAIFetcher,
    image_id: int,
    timeout: float = 30.0,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Look up a single image by ID and return its generation meta.

    CivitAI's `/images?imageId=X` endpoint wraps generation metadata
    in `{"id": X, "meta": <inner>}` (unlike the `modelVersionId`
    endpoint, which returns the inner dict directly). We unwrap here
    so callers see the same shape regardless of how the lookup was
    made.

    Returns (label, meta_or_None):
      - ('ok',         dict)  meta exists and has fields
      - ('no_meta',    None)  image exists but Civitai has no params
      - ('not_found',  None)  image record not in the API
      - ('error',      None)  network / parse failure
    """
    url = f"{fetcher.base_url}/images?imageId={image_id}"
    try:
        response = fetcher.session.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return ("not_found", None)
        return ("error", None)
    except Exception:
        return ("error", None)

    items = data.get("items") or []
    if not items:
        return ("not_found", None)

    outer = items[0].get("meta")
    # Detect the wrapper shape used by the imageId-lookup endpoint and
    # unwrap. The list endpoint returns inner directly, so we don't
    # over-unwrap if the shape is already inner-only.
    if (
        isinstance(outer, dict)
        and set(outer.keys()).issubset({"id", "meta"})
    ):
        inner = outer.get("meta")
    else:
        inner = outer

    if not isinstance(inner, dict) or not inner:
        return ("no_meta", None)
    return ("ok", inner)


def iter_target_files(
    media_root: Path, scope_glob: Optional[str]
) -> List[Tuple[Path, Path, int]]:
    """Collect (folder, file_path, image_id) for every numerically-
    named image file under media_root.

    The CivitAI download convention is `{folder}/{image_id}.{ext}`, so
    the numeric stem doubles as the API lookup key. Non-matching
    filenames (manual additions, our own .civitai-orig backups) are
    skipped silently.

    `scope_glob` is matched against the model-folder name (case
    sensitive), so `Plant Milk*` matches the Flax + Almond folders
    but nothing else.
    """
    out: List[Tuple[Path, Path, int]] = []
    for folder in sorted(media_root.iterdir()):
        if not folder.is_dir():
            continue
        if scope_glob and not fnmatch.fnmatch(folder.name, scope_glob):
            continue
        out.extend(_collect_from_folder(folder))
    return out


def _collect_from_folder(folder: Path) -> List[Tuple[Path, Path, int]]:
    """Collect (folder, file_path, image_id) for image files directly
    inside `folder`. Shared by the media-root and single-folder code
    paths so the file filter stays identical between them.
    """
    out: List[Tuple[Path, Path, int]] = []
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        m = _IMAGE_NAME_RE.match(f.name)
        if not m:
            continue
        out.append((folder, f, int(m.group(1))))
    return out


def _classify_file(path: Path) -> Tuple[str, str]:
    """Return (format_label, existing_params_string).

    format_label is one of 'png', 'jpeg', or 'unsupported'. The
    existing_params_string is the file's currently-embedded
    parameters text (or '' if none / unsupported).
    """
    from civitai_to_obsidian import (
        _detect_image_format,
        _read_jpeg_parameters,
        _read_png_parameters,
    )
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return ("unsupported", "")
    fmt = _detect_image_format(head)
    if fmt == "png":
        return ("png", _read_png_parameters(path))
    if fmt == "jpeg":
        return ("jpeg", _read_jpeg_parameters(path))
    return ("unsupported", "")


def is_complete(params_string: str) -> bool:
    """True if the parsed params cover the core A1111 fields.

    Used by --skip-complete to avoid spending an API call on files we
    already know are good enough.
    """
    if not params_string:
        return False
    parsed = parse_a1111_params(params_string)
    return _COMPLETENESS_KEYS.issubset(parsed.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill missing generation metadata in previously-"
            "downloaded CivitAI images. Dry-run by default; pass "
            "--apply to write."
        )
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually patch files. Without this, runs dry — reports "
             "what WOULD change but writes nothing.",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Before the first write to a given file, copy the "
             "original to `<name>.civitai-orig`. Subsequent writes "
             "to the same file leave the existing backup alone.",
    )
    parser.add_argument(
        "--scope", default=None,
        help='Glob pattern matched against folder names — restricts '
             'the scan to matching model folders. '
             'Example: --scope "Plant Milk*"',
    )
    parser.add_argument(
        "--folder", default=None,
        help="Process this single folder directly, ignoring config's "
             "media_folder. Use when you want to target one model's "
             "image folder without touching the rest. Mutually "
             "exclusive with --media-root and --scope.",
    )
    parser.add_argument(
        "--media-root", default=None,
        help="Override config's obsidian.media_folder to use this "
             "directory as the root to walk for model subfolders. "
             "Useful when testing against a copy of the vault.",
    )
    parser.add_argument(
        "--api-delay", type=float, default=1.5,
        help="Seconds to sleep between CivitAI API lookups "
             "(default: 1.5)",
    )
    parser.add_argument(
        "--skip-complete", action="store_true",
        help=(
            "Skip files whose embedded metadata already covers the "
            "core A1111 fields (prompt, Steps, Sampler, CFG scale, "
            "Seed, Size, Model). Faster — avoids API calls for files "
            "we already know are good. Default is to check every file."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Stop after processing this many files (0 = no limit). "
             "Useful for sanity-checking on a subset before a full "
             "run.",
    )
    args = parser.parse_args()

    # Validate the path-selection flags up front: they're mutually
    # exclusive because each represents a different intent (scan
    # everything / scan from a custom root / target one folder).
    if args.folder and (args.media_root or args.scope):
        print(
            "❌ --folder is mutually exclusive with --media-root and "
            "--scope. Pick one path-selection mode.",
            file=sys.stderr,
        )
        sys.exit(2)

    config = load_config(args.config)
    obs = config.get("obsidian", {})

    if args.folder:
        single_folder = Path(args.folder).expanduser().resolve()
        if not single_folder.is_dir():
            print(
                f"❌ --folder path is not a directory: {single_folder}",
                file=sys.stderr,
            )
            sys.exit(1)
        media_root = single_folder  # Used only for the display header
    elif args.media_root:
        media_root = Path(args.media_root).expanduser().resolve()
        if not media_root.is_dir():
            print(
                f"❌ --media-root is not a directory: {media_root}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        vault_path = Path(obs.get("vault_path", ".")).resolve()
        media_rel = obs.get(
            "media_folder", "zzMedia/Model and Lora Example Images"
        )
        media_root = (vault_path / media_rel).resolve()
        if not media_root.exists():
            print(
                f"❌ Media folder not found: {media_root}",
                file=sys.stderr,
            )
            sys.exit(1)

    api_key = config.get("civitai", {}).get("api_key")
    if not api_key:
        print(
            "⚠️  No CivitAI API key in config — anonymous requests "
            "are rate-limited and may 403 frequently.\n"
            "    Set civitai.api_key in config.yaml for best results."
        )

    fetcher = CivitAIFetcher(api_key=api_key)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"🔍 {mode}")
    if args.folder:
        print(f"   Single folder:  {media_root}")
    else:
        print(f"   Media root:     {media_root}")
        print(f"   Scope:          "
              f"{args.scope or '<all model folders>'}")
    print(f"   API delay:      {args.api_delay}s")
    print(f"   Skip complete:  {args.skip_complete}")
    if args.apply and args.backup:
        print(f"   Backups:        enabled (.civitai-orig)")
    print()

    if args.folder:
        files = _collect_from_folder(media_root)
    else:
        files = iter_target_files(media_root, args.scope)
    if not files:
        print("No matching image files found. Check --scope.")
        return
    if args.limit:
        files = files[:args.limit]
        print(f"⚠️  --limit {args.limit} — only processing first "
              f"{len(files)} file(s).\n")

    print(f"📂 {len(files)} candidate file(s) under "
          f"{'scope' if args.scope else 'media root'}\n")

    by_status: Counter[str] = Counter()
    per_folder: Dict[str, Counter[str]] = {}
    changes_preview: List[Tuple[str, str, List[str], List[str]]] = []
    error_samples: List[Tuple[str, str, str]] = []
    # Files that exercised the JSON-prompt rescue or Lora-hashes
    # synthesis paths. Surfaced separately in the report so the user
    # can audit every non-trivial decision the merger made.
    notable_samples: List[Tuple[str, str, List[str]]] = []

    api_calls_made = 0
    started = time.monotonic()
    last_progress = started

    for i, (folder, path, image_id) in enumerate(files, 1):
        folder_counter = per_folder.setdefault(folder.name, Counter())

        # Pre-flight: classify the file BEFORE we spend an API call.
        # Lets us skip unsupported formats (WEBP/GIF) and already-
        # complete files without burning quota.
        fmt_label, existing_params = _classify_file(path)
        if fmt_label == "unsupported":
            by_status[EnrichStatus.UNSUPPORTED] += 1
            folder_counter[EnrichStatus.UNSUPPORTED] += 1
            continue

        if args.skip_complete and is_complete(existing_params):
            by_status["skipped_complete"] += 1
            folder_counter["skipped_complete"] += 1
            continue

        api_label, api_meta = fetch_image_meta(fetcher, image_id)
        api_calls_made += 1

        if api_label != "ok":
            by_status[f"api_{api_label}"] += 1
            folder_counter[f"api_{api_label}"] += 1
        else:
            result = enrich_file(
                path,
                api_meta,
                dry_run=not args.apply,
                backup=args.backup,
            )
            by_status[result.status] += 1
            folder_counter[result.status] += 1

            if result.status == EnrichStatus.CHANGED:
                if len(changes_preview) < 15:
                    changes_preview.append(
                        (
                            folder.name,
                            path.name,
                            list(result.added_keys),
                            list(result.notes),
                        )
                    )
                # Always record files that triggered a special merge
                # decision, even past the 15-line preview cap — these
                # are the cases the user explicitly asked to see.
                if result.notes and len(notable_samples) < 40:
                    notable_samples.append(
                        (folder.name, path.name, list(result.notes))
                    )
            elif result.status == EnrichStatus.ERROR:
                msg = result.error or "<unknown>"
                if len(error_samples) < 10:
                    error_samples.append(
                        (folder.name, path.name, msg)
                    )

        # Only sleep when we actually hit the API. Saves a lot of
        # wall-clock time on big runs where most files are skipped.
        time.sleep(args.api_delay)

        # Heartbeat every 5s — long runs without output feel stuck.
        now = time.monotonic()
        if now - last_progress >= 5 or i == len(files):
            elapsed = now - started
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(files) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(files)}] last: {folder.name}/"
                f"{path.name} · api_calls={api_calls_made} · "
                f"eta {remaining:.0f}s"
            )
            last_progress = now

    # ---- Summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)

    total = len(files)
    changed = by_status.get(EnrichStatus.CHANGED, 0)
    no_change = by_status.get(EnrichStatus.NO_CHANGE, 0)
    embed_errors = by_status.get(EnrichStatus.ERROR, 0)
    unsupported = by_status.get(EnrichStatus.UNSUPPORTED, 0)
    not_found = by_status.get("api_not_found", 0)
    no_meta = by_status.get("api_no_meta", 0)
    api_errors = by_status.get("api_error", 0)
    skipped_complete = by_status.get("skipped_complete", 0)

    change_label = "Would change" if not args.apply else "Changed"
    print(f"  {change_label}:              {changed}")
    print(f"  Already complete (file):  {no_change}")
    if skipped_complete:
        print(
            f"  Skipped (--skip-complete):{skipped_complete}"
        )
    print(f"  API has no metadata:      {no_meta}")
    print(f"  Not found in API:         {not_found}")
    print(f"  Unsupported format:       {unsupported}")
    print(f"  API errors:               {api_errors}")
    print(f"  Embed errors:             {embed_errors}")
    print(f"  Total processed:          {total}")
    print(f"  API calls made:           {api_calls_made}")
    print(
        f"  Elapsed:                  "
        f"{time.monotonic() - started:.1f}s"
    )

    if changes_preview:
        print()
        verb = "Patched" if args.apply else "Would patch"
        print(f"📝 Sample — {verb} files (up to 15):")
        for folder_name, file_name, keys, notes in changes_preview:
            preview = ", ".join(keys[:5])
            if len(keys) > 5:
                preview += f", +{len(keys) - 5} more"
            print(f"   {folder_name}/{file_name} → +{preview}")
            for n in notes:
                print(f"      ⚡ {n}")

    if notable_samples:
        print()
        print(
            f"🔍 Files where a special merge decision was made "
            f"(up to {len(notable_samples)}):"
        )
        for folder_name, file_name, notes in notable_samples:
            print(f"   {folder_name}/{file_name}")
            for n in notes:
                print(f"      ⚡ {n}")

    if error_samples:
        print()
        print(f"⚠️  Embed errors (up to 10):")
        for folder_name, file_name, msg in error_samples:
            print(f"   {folder_name}/{file_name}: {msg}")

    # Per-folder breakdown when more than one folder was touched.
    if len(per_folder) > 1:
        print()
        print("📁 Per-folder breakdown (changed / no-change / skipped"
              " / errors):")
        for fname in sorted(per_folder):
            c = per_folder[fname]
            ch = c.get(EnrichStatus.CHANGED, 0)
            nc = c.get(EnrichStatus.NO_CHANGE, 0)
            sk = (
                c.get("skipped_complete", 0)
                + c.get("api_no_meta", 0)
                + c.get("api_not_found", 0)
                + c.get(EnrichStatus.UNSUPPORTED, 0)
            )
            er = c.get(EnrichStatus.ERROR, 0) + c.get("api_error", 0)
            print(
                f"   {fname}: ch={ch} nc={nc} sk={sk} er={er}"
            )

    if not args.apply:
        print(
            "\n💡 Dry run only. Re-run with --apply to write these "
            "changes."
        )


if __name__ == "__main__":
    main()
