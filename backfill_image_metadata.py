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
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
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


# Filename patterns we recognise as CivitAI downloads:
#   New convention: `{image_id}.{ext}`            e.g. 12345.png
#   Old convention: `{model_name}_{image_id}.{ext}`
#                                                e.g. Foo_Bar_12345.png
#
# Older downloads prefixed the filename with the folded folder name;
# newer downloads (post-ca7cdac folder-naming update) use the bare
# image id. The image id is always the trailing digit sequence
# before the extension, optionally preceded by a `_` or `-` separator.
# This lets the script process both naming conventions in the same
# tree without manual renaming.
#
# The `\d{4,}` floor (image id must be at least 4 digits) prevents
# wrong-lookup hazards where a trailing version-style suffix like
# `model_name_x_5.png` would otherwise be parsed as image id 5 — and
# the API lookup for id 5 would return SOME unrelated old image
# whose metadata would then get merged into the wrong file. Real
# CivitAI image ids are in the millions today (smallest observed in
# this user's library: 1.8M); 4 digits gives a 1000x safety margin
# without rejecting any plausibly-real downloads.
#
# `.civitai-orig` backup files don't match (extension isn't in the
# whitelist) and neither do hand-dropped files like `image20.png`
# (no separator before the digits).
_IMAGE_NAME_RE = re.compile(
    r"^(?:.*[_\-])?(\d{4,})\.(png|jpg|jpeg|webp|gif)$", re.IGNORECASE
)

# A file is treated as "already complete" — and skipped when
# --skip-complete is set — if its embedded params include all of these
# keys. They're the core A1111 fields most downstream tools care
# about. Civitai-specific extras (resources, hashes, etc.) aren't on
# the list because they're not in API meta to begin with.
_COMPLETENESS_KEYS = frozenset({
    "prompt", "Steps", "Sampler", "CFG scale", "Seed", "Size", "Model",
})

# Resumable-state machinery -------------------------------------------
#
# The script writes a JSON state file after every processed file so a
# long run can be interrupted (rate limits, ctrl-C, crash) and resumed
# without redoing already-processed files. A human-readable markdown
# doc is rendered alongside it for at-a-glance progress checks.
#
# Layout (defaults — both can be overridden by CLI flags):
#   <media_root>/.backfill-state.json    — machine-readable state
#   <media_root>/backfill-progress.md    — human-readable progress
#
# Skip rule: any file whose recorded status is in _TERMINAL_STATUSES
# is skipped on the next run. The distinction is intentional:
#   - `api_error` and embed `error` are TRANSPORT / WRITE failures
#     (transient network, partial download, temporary bug) — RETRY.
#   - `api_not_found` and `api_no_meta` are DETERMINISTIC API answers
#     (Civitai says "image doesn't exist" or "we have no params") —
#     these won't change on retry, so we treat them as terminal to
#     avoid burning the API quota in a re-run loop. They DO surface
#     in the markdown's Issues section if the user wants to revisit
#     them manually (e.g. by deleting their state entry).
#   - Other terminal statuses (CHANGED / NO_CHANGE / UNSUPPORTED /
#     skipped_complete) are stable end states.
# Add new error-like statuses to _ERROR_LIKE_STATUSES too, otherwise
# they won't surface in the Issues section even after a re-run.
_STATE_VERSION = 1
_DEFAULT_STATE_FILE = ".backfill-state.json"
_DEFAULT_PROGRESS_MD = "backfill-progress.md"
_TERMINAL_STATUSES = frozenset({
    EnrichStatus.CHANGED,
    EnrichStatus.NO_CHANGE,
    EnrichStatus.UNSUPPORTED,
    EnrichStatus.NOT_FOUND,
    EnrichStatus.NO_API_DATA,
    "api_not_found",
    "api_no_meta",
    "skipped_complete",
})
# Statuses worth flagging in the markdown's "Issues" section. Kept
# separate from _TERMINAL_STATUSES because the two concepts don't
# match: api_not_found is terminal but isn't an error to surface,
# while api_error is retryable but IS worth surfacing.
_ERROR_LIKE_STATUSES = frozenset({
    "api_error",
    EnrichStatus.ERROR,
})


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 with explicit Z. Keeps the state file
    parseable across machines without timezone guessing."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_key(folder_name: str, file_name: str) -> str:
    """Composite key for the state file. Including the folder name
    avoids collisions if two folders happen to contain the same image
    id (shared base file across model variants is plausible)."""
    return f"{folder_name}/{file_name}"


def _empty_state() -> Dict[str, Any]:
    now = _now_iso()
    return {
        "version": _STATE_VERSION,
        "started_at": now,
        "last_updated": now,
        "files": {},
    }


def load_state(path: Path) -> Dict[str, Any]:
    """Read the on-disk state file, or return a fresh structure if
    none exists yet. A corrupt, wrong-shaped, or wrong-version state
    file logs a warning and starts fresh — better than refusing to
    run on an unreadable resume point, and safer than silently
    mutating a state file written by an incompatible schema.

    Narrow exception list on purpose: I/O errors, JSON parse errors,
    and decode errors are the documented failure modes for json.load
    on a text file. A broader `except Exception` would mask coding
    bugs (AttributeError, TypeError) that should fail loudly instead.
    """
    if not path.exists():
        return _empty_state()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(
            f"⚠️  Could not read state file {path} ({exc}); "
            f"starting fresh."
        )
        return _empty_state()
    if not isinstance(data, dict) or "files" not in data:
        print(
            f"⚠️  State file {path} has unexpected shape; "
            f"starting fresh."
        )
        return _empty_state()
    file_version = data.get("version")
    if file_version != _STATE_VERSION:
        # Schema mismatch — could be a state file written by a
        # newer or older copy of this script. Starting fresh is
        # safer than guessing whether the entry layout is
        # compatible.
        print(
            f"⚠️  State file {path} has version "
            f"{file_version!r}, expected {_STATE_VERSION}; "
            f"starting fresh."
        )
        return _empty_state()
    return data


def save_state(path: Path, state: Dict[str, Any]) -> None:
    """Atomic write so a crash mid-write leaves the previous good
    state intact rather than a half-written JSON.

    Uses a unique tempfile in the same directory (NamedTemporaryFile
    with delete=False) so:
      - The final os.replace is a same-filesystem rename (atomic).
      - Two concurrent writers don't clobber each other's tempfile.
      - A leftover tempfile from a crashed prior run isn't silently
        overwritten by `open(..., "w")`.
    """
    state["last_updated"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            delete=False,
            prefix=path.name + ".",
            suffix=".tmp",
            encoding="utf-8",
        ) as tf:
            json.dump(state, tf, indent=2, sort_keys=True)
            tmp_path = Path(tf.name)
        os.replace(tmp_path, path)
    except OSError:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def compute_expected_per_folder(
    media_root: Path,
    single_folder: bool,
) -> Dict[str, int]:
    """Count target image files per folder so the markdown can show
    'X / Y processed' even before processing starts.

    In single-folder mode, media_root IS the folder to walk, and we
    key it by its own name. Otherwise, walk every subfolder of
    media_root — intentionally ignoring scope so the markdown shows
    the FULL tree's state, not just the slice of a scoped run. A
    scoped run leaves other folders untouched but the user still
    needs to see them as 'pending' in the checklist.

    Folders with zero matching files are omitted — these are
    typically container directories (like `image-backups/`) that
    shouldn't appear in the progress checklist.
    """
    out: Dict[str, int] = {}
    if single_folder:
        n = sum(
            1 for f in media_root.iterdir()
            if f.is_file() and _IMAGE_NAME_RE.match(f.name)
        )
        if n > 0:
            out[media_root.name] = n
        return out
    for folder in sorted(media_root.iterdir()):
        if not folder.is_dir():
            continue
        n = sum(
            1 for f in folder.iterdir()
            if f.is_file() and _IMAGE_NAME_RE.match(f.name)
        )
        if n > 0:
            out[folder.name] = n
    return out


def _folder_counts(state: Dict[str, Any]) -> Dict[str, Counter]:
    """Bucket per-file state entries into per-folder Counters."""
    per_folder: Dict[str, Counter] = defaultdict(Counter)
    for key, entry in state.get("files", {}).items():
        if "/" not in key:
            continue
        folder_name = key.split("/", 1)[0]
        status = entry.get("status", "unknown")
        per_folder[folder_name][status] += 1
    return per_folder


def render_progress_md(
    state: Dict[str, Any],
    expected_per_folder: Dict[str, int],
    media_root: Path,
) -> str:
    """Build the human-readable progress doc. The checklist is the
    primary signal — [x] folders are done, [ ] are pending or partial.
    Errors get a separate section so they don't get lost."""
    per_folder = _folder_counts(state)

    total_expected = sum(expected_per_folder.values())
    total_processed = sum(sum(c.values()) for c in per_folder.values())

    folders_complete = 0
    for fname, expected in expected_per_folder.items():
        processed = sum(per_folder.get(fname, Counter()).values())
        if expected > 0 and processed >= expected:
            folders_complete += 1
    folders_total = len(expected_per_folder)

    lines: List[str] = [
        "# Backfill Progress",
        "",
        f"**Media root:** `{media_root}`  ",
        f"**Last updated:** {state.get('last_updated', '?')}  ",
        f"**Started:** {state.get('started_at', '?')}",
        "",
        "## Summary",
        "",
        f"- Folders complete: **{folders_complete} / "
        f"{folders_total}**",
        f"- Files processed: **{total_processed} / "
        f"{total_expected}**",
        "",
        "## Folders",
        "",
    ]

    # Order: pending/partial first (work to do), then completed.
    # Within each group, sort by folder name. Makes it easy to scroll
    # to "what's left" on a long list.
    pending_lines: List[str] = []
    done_lines: List[str] = []
    for fname in sorted(expected_per_folder):
        expected = expected_per_folder[fname]
        counts = per_folder.get(fname, Counter())
        processed = sum(counts.values())
        done = expected > 0 and processed >= expected
        if done:
            done_lines.append(_render_folder_line(fname, expected,
                                                  processed, counts,
                                                  True))
        else:
            pending_lines.append(_render_folder_line(fname, expected,
                                                     processed, counts,
                                                     False))

    if pending_lines:
        lines.append("### Pending / In progress")
        lines.append("")
        lines.extend(pending_lines)
        lines.append("")
    if done_lines:
        lines.append("### Completed")
        lines.append("")
        lines.extend(done_lines)
        lines.append("")

    # Surface error files so they're not buried inside per-folder
    # counts. These are the ones worth re-running or investigating.
    error_entries = [
        (k, v.get("status", "unknown"))
        for k, v in state.get("files", {}).items()
        if v.get("status") in _ERROR_LIKE_STATUSES
    ]
    if error_entries:
        lines.extend(["## Issues", ""])
        for key, status in error_entries[:100]:
            lines.append(f"- `{key}` — {status}")
        if len(error_entries) > 100:
            lines.append(
                f"- ...and {len(error_entries) - 100} more"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_folder_line(
    fname: str,
    expected: int,
    processed: int,
    counts: Counter,
    done: bool,
) -> str:
    """Format a single folder's checklist row with status breakdown."""
    mark = "x" if done else " "
    if processed == 0:
        detail = f"pending ({expected} files)"
    elif done:
        # Show the breakdown so the user can spot folders that
        # finished but had unusual outcomes (lots of no_meta, etc.).
        parts = [
            f"{status}: {count}"
            for status, count in sorted(counts.items())
            if count > 0
        ]
        detail = f"{processed} files — {', '.join(parts)}"
    else:
        detail = f"{processed} / {expected} processed"
    return f"- [{mark}] **{fname}** — {detail}"


def save_progress_md(
    path: Path,
    state: Dict[str, Any],
    expected_per_folder: Dict[str, int],
    media_root: Path,
) -> None:
    """Render and write the markdown progress doc. Not atomic
    because partial markdown is harmless (next write replaces it),
    and we'd rather avoid the .tmp file showing up in the user's
    folder listing.

    Explicit UTF-8 encoding because folder names routinely contain
    emoji and other non-ASCII characters; relying on locale-default
    encoding would crash on non-UTF-8 locales (e.g. LANG=C)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_progress_md(state, expected_per_folder,
                                    media_root))


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
    """Collect (folder, file_path, image_id) for every recognised
    CivitAI image file under media_root.

    Two filename conventions are accepted (see _IMAGE_NAME_RE):
      - bare `{image_id}.{ext}` (newer downloads)
      - `{folder_name}_{image_id}.{ext}` (older downloads — the
        folded folder name was prefixed for disambiguation)
    The trailing digit sequence is the API lookup key. Non-matching
    filenames (manual additions, `.civitai-orig` backups, etc.) are
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
        "--backup-dir", default=None,
        help="Directory under which `.civitai-orig` backups are "
             "written, with one subfolder per source folder "
             "(<backup-dir>/<folder-name>/<file>.civitai-orig). "
             "Keeps source folders clean. Without this, backups land "
             "as siblings next to the original. Only takes effect "
             "when --backup is also passed (this flag specifies "
             "WHERE backups go, not WHETHER to create them). Can "
             "also be set via `backfill.backup_dir` in config.",
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
    parser.add_argument(
        "--state-file", default=None,
        help="Path to the resumable JSON state file. Records each "
             "file's outcome so a re-run picks up where it left off "
             "(skipping files with a terminal status). Default: "
             f"`{_DEFAULT_STATE_FILE}` inside the media root. Only "
             "written when --apply is set.",
    )
    parser.add_argument(
        "--progress-md", default=None,
        help="Path to the human-readable progress markdown file. "
             "Folder-level checklist updated during the run. "
             f"Default: `{_DEFAULT_PROGRESS_MD}` inside the media "
             "root.",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Delete any existing state file and progress markdown "
             "before starting — forces a full re-run from scratch. "
             "Without this, --apply runs resume from prior state.",
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

    # Resolve backup-dir: CLI flag wins, then config's
    # backfill.backup_dir, otherwise None (sibling-file backups).
    # backup_dir only specifies LOCATION — the user still has to
    # pass --backup explicitly to actually create backups. This
    # keeps backup opt-in: a configured location won't quietly
    # double disk usage on every run.
    backup_dir_raw = args.backup_dir or (
        config.get("backfill", {}).get("backup_dir")
    )
    backup_dir: Optional[Path] = None
    if backup_dir_raw:
        backup_dir = Path(backup_dir_raw).expanduser().resolve()

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

    # Resolve state + progress paths and initialise state. The state
    # file is only written when --apply is set (dry runs shouldn't
    # contaminate the resume point). The markdown is still rendered
    # on dry runs so the user can preview what the doc will look
    # like — it just reflects whatever's already persisted.
    state_file_path = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else media_root / _DEFAULT_STATE_FILE
    )
    progress_md_path = (
        Path(args.progress_md).expanduser().resolve()
        if args.progress_md
        else media_root / _DEFAULT_PROGRESS_MD
    )
    if args.restart:
        for p in (state_file_path, progress_md_path):
            if p.exists():
                p.unlink()
                print(f"🗑️  Removed {p}")
    state = load_state(state_file_path)

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
    if args.apply:
        print(f"   State file:     {state_file_path}")
    print(f"   Progress doc:   {progress_md_path}")
    if args.apply and args.backup:
        if backup_dir is not None:
            print(f"   Backups:        enabled → {backup_dir}/"
                  f"<folder>/<file>.civitai-orig")
        else:
            print(f"   Backups:        enabled (sibling "
                  f".civitai-orig)")
    elif args.apply:
        print(f"   Backups:        disabled (pass --backup to enable)")
    print()

    expected_per_folder = compute_expected_per_folder(
        media_root, bool(args.folder)
    )

    prior_skip_count = sum(
        1 for entry in state.get("files", {}).values()
        if entry.get("status") in _TERMINAL_STATUSES
    )
    if args.apply and prior_skip_count:
        print(
            f"♻️  Resuming from state file — {prior_skip_count} "
            f"file(s) already have a terminal status and will be "
            f"skipped."
        )

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
    resumed_skipped = 0
    started = time.monotonic()
    last_progress = started
    last_md_save = started
    last_folder_name: Optional[str] = None
    md_save_interval = 30.0  # seconds — keep the doc readable mid-run

    # Initial markdown render (apply runs only) so it exists from
    # the start, even if the run is interrupted before the first
    # file lands. Dry runs don't touch persisted artifacts — they're
    # preview-only.
    if args.apply:
        save_progress_md(progress_md_path, state, expected_per_folder,
                         media_root)

    for i, (folder, path, image_id) in enumerate(files, 1):
        folder_counter = per_folder.setdefault(folder.name, Counter())

        # Refresh markdown when crossing a folder boundary — this is
        # the natural unit for the progress doc and the moment the
        # user is most likely to want a fresh view.
        if (args.apply and last_folder_name is not None
                and last_folder_name != folder.name):
            save_progress_md(progress_md_path, state,
                             expected_per_folder, media_root)
            last_md_save = time.monotonic()
        last_folder_name = folder.name

        # Resume short-circuit: if this file already has a terminal
        # status from a prior --apply run, count it locally for the
        # session summary but skip all work (no classify, no API,
        # no sleep). Only honoured when --apply is set so dry runs
        # don't silently skip files the user wanted to preview.
        key = _state_key(folder.name, path.name)
        prior = state["files"].get(key) if args.apply else None
        if prior and prior.get("status") in _TERMINAL_STATUSES:
            prior_status = prior["status"]
            by_status[prior_status] += 1
            folder_counter[prior_status] += 1
            resumed_skipped += 1
            continue

        # Pre-flight: classify the file BEFORE we spend an API call.
        # Lets us skip unsupported formats (WEBP/GIF) and already-
        # complete files without burning quota.
        final_status: Optional[str] = None
        fmt_label, existing_params = _classify_file(path)
        if fmt_label == "unsupported":
            by_status[EnrichStatus.UNSUPPORTED] += 1
            folder_counter[EnrichStatus.UNSUPPORTED] += 1
            final_status = EnrichStatus.UNSUPPORTED
        elif args.skip_complete and is_complete(existing_params):
            by_status["skipped_complete"] += 1
            folder_counter["skipped_complete"] += 1
            final_status = "skipped_complete"
        else:
            api_label, api_meta = fetch_image_meta(fetcher, image_id)
            api_calls_made += 1

            if api_label != "ok":
                status_key = f"api_{api_label}"
                by_status[status_key] += 1
                folder_counter[status_key] += 1
                final_status = status_key
            else:
                result = enrich_file(
                    path,
                    api_meta,
                    dry_run=not args.apply,
                    backup=args.backup,
                    backup_dir=backup_dir,
                )
                by_status[result.status] += 1
                folder_counter[result.status] += 1
                final_status = result.status

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
                    # Always record files that triggered a special
                    # merge decision, even past the 15-line preview
                    # cap — these are the cases the user explicitly
                    # asked to see.
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

            # Only sleep when we actually hit the API. Saves a lot
            # of wall-clock time on big runs where most files are
            # skipped via classify/skip-complete paths.
            time.sleep(args.api_delay)

        # Persist state after every processed file (--apply only).
        # Atomic write keeps the resume point safe against crashes.
        if args.apply and final_status is not None:
            state["files"][key] = {
                "status": final_status,
                "ts": _now_iso(),
            }
            save_state(state_file_path, state)

        # Throttled markdown refresh so monitoring from another
        # terminal stays useful inside large folders.
        now = time.monotonic()
        if args.apply and now - last_md_save >= md_save_interval:
            save_progress_md(progress_md_path, state,
                             expected_per_folder, media_root)
            last_md_save = now

        # Heartbeat every 5s — long runs without output feel stuck.
        if now - last_progress >= 5 or i == len(files):
            elapsed = now - started
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(files) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(files)}] last: {folder.name}/"
                f"{path.name} · api_calls={api_calls_made} · "
                f"resumed_skip={resumed_skipped} · "
                f"eta {remaining:.0f}s"
            )
            last_progress = now

    # Final markdown render so the doc reflects the end state.
    if args.apply:
        save_progress_md(progress_md_path, state, expected_per_folder,
                         media_root)

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
    if resumed_skipped:
        print(
            f"  Resumed-skip (state):     {resumed_skipped}"
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
    print(f"  Progress doc:             {progress_md_path}")
    if args.apply:
        print(f"  State file:               {state_file_path}")

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
