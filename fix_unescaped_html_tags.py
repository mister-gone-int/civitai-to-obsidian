#!/usr/bin/env python3
"""
Backslash-escape HTML-tag-shaped `<...>` patterns sitting outside
fenced code blocks in model markdown files.

Civitai metadata values such as `triggerWords: ['<SignWithtextINeedBuzz>']`
get emitted as plain-markdown list items. CommonMark recognizes
`<tagname...>` (where tagname = letter + [A-Za-z0-9-]*) as raw inline
HTML. Obsidian's Live Preview hands those to the HTML parser, which
treats the unknown element as an unclosed container and swallows
every following line — image embeds stop rendering and formatting
collapses from that point on.

This script finds every such pattern outside fenced (```) code blocks
and rewrites the angle brackets as `\\<...\\>`, which renders as
literal `<...>` in Obsidian without triggering HTML parsing. Content
inside fenced code blocks is never touched. Patterns whose names are
not valid HTML tag names (e.g. `<lora:foo>`) are also untouched —
they don't trigger the bug.

Defaults to a dry run. Pass --apply to actually rewrite files.

Usage:
    python fix_unescaped_html_tags.py                   # dry run
    python fix_unescaped_html_tags.py --apply           # do it
    python fix_unescaped_html_tags.py --config foo.yaml # alt config
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from civitai_to_obsidian import load_config
from fix_existing_images import (
    backup_files,
    iter_markdown_files,
    write_atomic,
)

# Matches CommonMark's "open tag" / "closing tag" raw-HTML inline
# patterns: `<` + optional `/` + tagname + optional attributes (no
# nested angles) + optional `/` + `>`. This is intentionally a strict
# match — anything that wouldn't trip the HTML parser is left alone.
_RAW_HTML_TAG_RE = re.compile(
    r"<(/?[A-Za-z][A-Za-z0-9-]*)((?:\s[^<>]*)?)(/?)>"
)

_FENCE_RE = re.compile(r"^\s*```")


def _escape_tags(line: str) -> str:
    """Backslash-escape every CommonMark tag-shaped `<...>` on a line.
    `\\<` and `\\>` render as literal `<` and `>` in Obsidian."""
    return _RAW_HTML_TAG_RE.sub(
        lambda m: "\\<" + m.group(1) + m.group(2) + m.group(3) + "\\>",
        line,
    )


def rewrite_file(text: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Return (new_text, hits). `hits` is a list of
    (line_number_1_based, old_line, new_line) for every line that
    changed.

    Tracks fenced code block state by toggling on lines that begin
    with three or more backticks (after optional whitespace) — this
    matches CommonMark's fence rules well enough for the metadata
    docs this project produces.
    """
    new_lines: List[str] = []
    hits: List[Tuple[int, str, str]] = []
    in_fence = False
    for i, line in enumerate(text.splitlines(keepends=True), start=1):
        stripped = line.rstrip("\n")
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
            new_lines.append(line)
            continue
        if in_fence:
            new_lines.append(line)
            continue
        if "<" not in stripped:
            new_lines.append(line)
            continue
        ending = line[len(stripped):]  # preserve trailing \n / \r\n
        replaced = _escape_tags(stripped)
        if replaced != stripped:
            hits.append((i, stripped, replaced))
        new_lines.append(replaced + ending)
    return "".join(new_lines), hits


def check_rewrite_invariants(old: str, new: str) -> str:
    """Return '' if structural invariants hold, else a reason string.

    We only modify in-line `<` / `>` characters. Line count, embed
    count, and code-fence count must all be preserved — if any of
    those change, the rewrite is suspect and we skip the file.
    """
    if old.count("\n") != new.count("\n"):
        return (
            f"line count changed ({old.count(chr(10))} → "
            f"{new.count(chr(10))})"
        )
    if old.count("![[") != new.count("![["):
        return f"![[ count changed ({old.count('![[')} → {new.count('![[')})"
    if old.count("]]") != new.count("]]"):
        return f"]] count changed ({old.count(']]')} → {new.count(']]')})"
    if old.count("```") != new.count("```"):
        return (
            f"code-fence count changed ({old.count('```')} → "
            f"{new.count('```')})"
        )
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Escape unescaped HTML-tag-shaped patterns in model "
            "markdown files. Default is dry run."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually rewrite files. Without this flag, runs as a "
            "dry run that only reports what would change."
        ),
    )
    parser.add_argument(
        "--scan-all-markdown",
        action="store_true",
        help=(
            "Scan the entire vault instead of just "
            "`obsidian.base_directory`."
        ),
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help=(
            "Where to store backups of touched markdown. Defaults to "
            "`<vault>/.civitai-htmlfix-backup-<timestamp>/`."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    obs = config.get("obsidian", {})
    vault_path = Path(obs.get("vault_path", ".")).resolve()
    base_dir_rel = obs.get("base_directory", "")
    base_dir = (
        (vault_path / base_dir_rel).resolve() if base_dir_rel else None
    )

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
            file=sys.stderr,
        )
        return 1

    if args.backup_dir:
        backup_root = Path(args.backup_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_root = vault_path / f".civitai-htmlfix-backup-{stamp}"

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"🔍 {mode}")
    print(f"   Markdown scope: {scope_label}")
    print(f"   Backup dir:     {backup_root}\n")

    md_files = list(iter_markdown_files(md_roots))
    print(f"  Scanning {len(md_files)} markdown file(s)...\n")

    files_to_touch: List[Path] = []
    file_hits: List[Tuple[Path, str, List[Tuple[int, str, str]]]] = []
    total_hits = 0
    skipped: List[Tuple[Path, str]] = []

    for md_path in md_files:
        try:
            text = md_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_text, hits = rewrite_file(text)
        if not hits:
            continue
        problem = check_rewrite_invariants(text, new_text)
        if problem:
            skipped.append((md_path, problem))
            continue
        files_to_touch.append(md_path)
        file_hits.append((md_path, new_text, hits))
        total_hits += len(hits)

    if not files_to_touch:
        print("✅ Nothing to do — no unescaped tag-shaped patterns found.")
        return 0

    print(
        f"  Files needing repair: {len(files_to_touch)}  "
        f"(total occurrences: {total_hits})\n"
    )
    for md_path, _new_text, hits in file_hits:
        try:
            rel = md_path.relative_to(vault_path)
        except ValueError:
            rel = md_path
        print(f"  📝 {rel}")
        for lineno, old, new in hits[:5]:
            print(f"     L{lineno}: {old.strip()[:120]}")
            print(f"        → {new.strip()[:120]}")
        if len(hits) > 5:
            print(f"     ... and {len(hits) - 5} more occurrence(s)")
    print()

    for md_path, reason in skipped:
        print(f"  ⚠️  Skipped {md_path}: {reason}")
    if skipped:
        print()

    if not args.apply:
        print(
            "✋ Dry run — no files modified. Re-run with --apply to "
            "rewrite."
        )
        return 0

    backup_count = backup_files(files_to_touch, vault_path, backup_root)
    print(
        f"💾 Backed up {backup_count} markdown file(s) to "
        f"{backup_root}"
    )

    written = 0
    for md_path, new_text, _hits in file_hits:
        write_atomic(md_path, new_text)
        written += 1
    print(f"✅ Rewrote {written} file(s).")
    print(f"   Backups written to: {backup_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
