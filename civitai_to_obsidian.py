#!/usr/bin/env python3
"""
CivitAI to Obsidian Image Library Builder

This script fetches example images from CivitAI models/LoRAs along
with their generation parameters and creates a comprehensive Obsidian
markdown page.

Usage:
    python civitai_to_obsidian.py <model_url_or_id> [options]

Example:
    python civitai_to_obsidian.py https://civitai.com/models/12345
    python civitai_to_obsidian.py 12345 --limit 300

By: Kevin Neblett

"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Pillow + piexif power the metadata embedding helpers below. Both are
# declared in requirements.txt — import errors here mean the user is
# running an out-of-date environment and should `pip install -r
# requirements.txt`. We surface that clearly rather than crashing deep
# inside an embed call.
try:
    from PIL import Image, UnidentifiedImageError
    import piexif
    import piexif.helper
except ImportError as _imerr:  # pragma: no cover — install-state check
    print(
        "❌ Missing image-handling dependency:",
        _imerr,
        "\n   Run: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise

# piexif 1.1.3 has a TIFF-spec compliance bug in its sub-IFD writer
# (no 4-byte next-IFD pointer between entries and value payloads).
# Strict EXIF readers reject the output. We patch the writer at
# import time so every code path that calls piexif.dump produces
# compliant bytes. See _exif_writer.py for the full rationale.
from _exif_writer import (
    install_piexif_patches,
    splice_exif_app1,
    validate_subifd_compliance,
    ExifSpliceError,
)
install_piexif_patches()

# PNG repair lives in its own module so the same logic is reusable
# from both the download path and any future on-disk repair tooling.
# See _png_repair.py for the rationale (CivitAI CDN occasionally
# serves PNGs with a bad chunk CRC even though the content is intact).
from _png_repair import repair_png_bytes

# Interactive mode is opt-in via --interactive; keep the import soft so a
# user who never touches that flag doesn't need questionary installed.
try:
    import questionary
except ImportError:  # pragma: no cover — exercised by the install-check path
    questionary = None  # type: ignore[assignment]


# CivitAI's tipping currency is called "Buzz", and a steady stream of
# low-effort images get posted with prompts like "buzz please" or
# "give me buzz" in an attempt to farm tips. These patterns target
# that genre. They're case-insensitive and word-boundary aware so
# they don't catch legitimate uses like "buzzcut" or "fuzz".
DEFAULT_BEGGING_PATTERNS: List[str] = [
    # "buzz please/pls/me/up/appreciated/welcome/thanks" — bare buzz
    # paired with a begging cue. Allows optional punctuation between
    # the two words (e.g. "buzz, please").
    r'\bbuzz\s*[,.\-!?]*\s*'
    r'(please|pls|plz|me|up|appreciated|welcome|thanks|thx|ty)\b',
    # "please [give|send|share|tip|drop|spare] buzz" — please before
    # the optional verb; the verb is optional so "please buzz" hits.
    r'\bplease\s+(give|send|share|tip|drop|spare)?\s*'
    r'(some\s+|a\s+|me\s+)?buzz\b',
    # "{need|gimme|give me|send|send me|spare|drop} buzz" with optional
    # filler ("some buzz", "a buzz", "more buzz").
    r'\b(need|gimme|give\s+me|send(?:\s+me)?|spare|drop)\s+'
    r'(some\s+|a\s+|the\s+|me\s+|more\s+|any\s+)?buzz\b',
    # "{yellow|blue|green} buzz appreciated/please/etc" — color-prefixed
    # buzz tiers showing up in begging captions.
    r'\b(yellow|blue|green)\s+buzz\s+'
    r'(please|appreciated|welcome|pls|plz|tips?|tipping|thanks)\b',
    # Hashtag begging.
    r'#buzz\s*(farm|farming|please|pls|plz|me|tips?|tipping)\b',
    # "support me/us/this with/via buzz" — the explicit ask.
    r'\bsupport\s+(me|us|this|the\s+\w+)\s+'
    r'(with|via|by|using)\s+buzz\b',
]


def compile_begging_patterns(
    patterns: List[str]
) -> List[re.Pattern[str]]:
    """Compile a list of regex strings, dropping any that fail.

    A bad pattern from user config shouldn't crash the whole run —
    we warn and continue with the patterns that did compile, so the
    filter still does useful work.
    """
    compiled: List[re.Pattern[str]] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw, re.IGNORECASE))
        except re.error as exc:
            print(
                f"⚠️  Skipping invalid begging pattern {raw!r}: {exc}"
            )
    return compiled


def detect_begging_match(
    image_data: Dict[str, Any],
    patterns: List[re.Pattern[str]]
) -> Optional[str]:
    """Return the source of the first matching pattern, or None.

    Scans the prompt and negative prompt — that's where this stuff
    overwhelmingly lives, because users tack the beg onto the
    generation prompt so it travels with the image metadata. Other
    meta fields are ignored to keep false positives down.
    """
    meta = image_data.get("meta")
    if not isinstance(meta, dict):
        return None

    haystack_parts: List[str] = []
    for key in ("prompt", "negativePrompt"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            haystack_parts.append(value)

    if not haystack_parts:
        return None

    haystack = " \n ".join(haystack_parts)
    for pattern in patterns:
        if pattern.search(haystack):
            return pattern.pattern
    return None


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce a config value to a real bool, accepting YAML's many shapes.

    YAML loaders return real booleans for unquoted `true`/`false`, but
    users routinely quote them ("true") or use yes/no/on/off variants
    that come back as strings. Without coercion these flow into
    questionary.confirm's `default=` parameter as truthy non-bools and
    produce confusing prompt behavior — and elsewhere they sneak into
    `if config_value:` branches that were meant to short-circuit on
    literal False but never do.

    Anything that isn't recognizably truthy or falsy falls back to the
    caller-supplied default, so a malformed entry behaves the same as a
    missing entry.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('true', 'yes', 'y', '1', 'on'):
            return True
        if v in ('false', 'no', 'n', '0', 'off', ''):
            return False
    return default


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        print(
            f"❌ Config file not found: {config_path}\n"
            "Please copy config.example.yaml to config.yaml "
            "and customize it."
        )
        sys.exit(1)

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Generation-metadata embedding
#
# The download pipeline writes the raw bytes that CivitAI's CDN serves.
# Sometimes those bytes lack the embedded generation parameters even
# when the API has them — most often because the CDN-served file was
# re-encoded server-side without copying the EXIF/tEXt across. The
# helpers below let us:
#
#   1. Render CivitAI's `meta` dict to the standard A1111 "parameters"
#      string that A1111 / Forge / ComfyUI / Civitai's own viewer read.
#   2. Read whatever generation params are already embedded.
#   3. Merge the two non-destructively (file always wins on conflict;
#      API only fills gaps).
#   4. Write the merged string back into the file using the LEAST
#      invasive mechanism for that format:
#        - PNG  → manual tEXt chunk surgery (pixel data never decoded)
#        - JPEG → piexif.insert (only the EXIF APP1 segment changes;
#                 the compressed image data is byte-for-byte identical)
#        - WEBP → skipped (safe in-place EXIF patching needs raw RIFF
#                 chunk manipulation; deferred — see backfill script)
#
# Every write goes through a tempfile + verify + atomic rename, so a
# crash or assertion failure mid-write never leaves a damaged file.
# ---------------------------------------------------------------------

# Civitai's meta uses a mix of camelCase (prompt, negativePrompt,
# steps, sampler, cfgScale, seed, clipSkip, denoise, scheduler,
# resources, civitaiResources) and A1111-style Title Case (Model,
# Model hash, Clip skip, Size, Denoising strength, Hires upscale,
# Schedule type, Version, VAE). The canonical form for the A1111
# "parameters" string is Title Case, so we translate camelCase keys
# on the way in. Anything not in this map is passed through unchanged
# — that covers Civitai's already-Title-Case keys plus any forward-
# compat additions (aspectRatio, baseModel, models, vaes, upscalers,
# extra, hashes, workflow, process, etc.).
#
# The entries here are MANDATORY for non-duplication: without them,
# a file with `Clip skip: 2` and an API response with `clipSkip: 2`
# would merge into a file that has BOTH keys — exactly the duplication
# we're meant to prevent.
_CIVITAI_TO_A1111: Dict[str, str] = {
    # Standard A1111 prompt/param synonyms
    "negativePrompt": "Negative prompt",
    "steps": "Steps",
    "sampler": "Sampler",
    "cfgScale": "CFG scale",
    "seed": "Seed",
    "clipSkip": "Clip skip",
    "denoise": "Denoising strength",
    "scheduler": "Schedule type",
    # Resource lists — older (`resources`: {name, hash, type}) and
    # newer (`civitaiResources`: {type, modelVersionId, weight})
    # both collapse to the single `Civitai resources` key that
    # Civitai's own writer uses in the PNG `parameters` string.
    # An image typically has only one of the two, so this won't
    # collide within a single response.
    "resources": "Civitai resources",
    "civitaiResources": "Civitai resources",
}

# Keys we deliberately never write into the parameters string.
#   - `comfy`: full ComfyUI workflow JSON dump (kilobytes of nested
#     structure). ComfyUI carries it in its own `workflow` PNG chunk;
#     putting it on the A1111 params line bloats files and breaks
#     params-line parsers in some readers.
#   - `width`/`height`: redundant with `Size: WIDTHxHEIGHT`. We
#     synthesize Size from them during normalization and drop them.
#   - `prompt`/`Negative prompt`: special-cased into their own lines
#     by the formatter, not the params line.
_PARAMS_SKIP_KEYS = {
    "prompt", "negativePrompt", "Negative prompt",
    "comfy",
    "width", "height",
}

# Order that A1111/Forge emit known params in. Anything not listed
# here is appended in whatever order it appears in the source dict.
# `Lora hashes` is on the list so when we synthesize it we slot it
# in the canonical A1111 position rather than appending at the end.
_A1111_STANDARD_ORDER: List[str] = [
    "Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size",
    "Model hash", "Model", "VAE", "Denoising strength", "Clip skip",
    "Hires upscale", "Hires upscaler", "Hires steps", "Lora hashes",
    "Version",
]


def _normalize_meta(meta: Any) -> Dict[str, Any]:
    """Convert a Civitai meta dict (or any A1111-keyed dict) to the
    canonical Title-Case form used internally.

    Steps:
      1. Translate known camelCase synonyms (clipSkip → Clip skip,
         resources → Civitai resources, etc.) via _CIVITAI_TO_A1111.
      2. Drop empty values (None / blank strings) — they carry no
         information and would just add noise to the merge.
      3. Drop _PARAMS_SKIP_KEYS that aren't relevant (comfy workflow
         dump, redundant width/height).
      4. Synthesize `Size: WIDTHxHEIGHT` from `width`+`height` when
         present and `Size` isn't — A1111-style readers expect Size,
         and carrying width/height separately is a duplicate
         representation of the same fact.
    """
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        # Translate camelCase → A1111 canonical Title Case.
        norm_k = _CIVITAI_TO_A1111.get(k, k)
        # Skip ComfyUI workflow dump and the width/height keys we'll
        # synthesize below; keep prompt / Negative prompt so the
        # formatter can emit them as their own lines.
        if norm_k in _PARAMS_SKIP_KEYS - {"prompt", "Negative prompt"}:
            continue
        out[norm_k] = v

    # Synthesize Size from width+height. We read these from the raw
    # input (they were skipped above) so a caller passing only
    # width/height still gets a usable Size. Only fills the slot
    # when Size isn't already present — file-side Size always wins.
    if "Size" not in out:
        w = meta.get("width")
        h = meta.get("height")
        if (
            isinstance(w, (int, float, str))
            and isinstance(h, (int, float, str))
            and str(w).strip()
            and str(h).strip()
        ):
            # Normalize numeric strings: 832.0 → "832", "832" → "832"
            def _intify(x: Any) -> str:
                try:
                    n = float(x)
                    if n == int(n):
                        return str(int(n))
                except (TypeError, ValueError):
                    pass
                return str(x).strip()
            out["Size"] = f"{_intify(w)}x{_intify(h)}"

    return out


def _format_param_value(v: Any) -> str:
    """Render a value for the comma-separated params line.

    - dicts and lists → JSON-encoded (compact). Civitai's own writer
      does this for `Civitai resources` and `Civitai metadata`. Our
      bracket-aware parser round-trips JSON losslessly without
      needing quotes, and that matches the wire format SagaSigil /
      A1111 / Forge expect.
    - bracketed strings → emitted bare (caller already JSON-encoded,
      or read a JSON value back through parse_a1111_params).
    - strings with commas → wrapped in double quotes so the params-
      line splitter doesn't mis-split them. Embedded quotes escaped.
    - everything else → str() unchanged.

    None values should be filtered upstream by _normalize_meta; if
    one slips through here it becomes the literal text "None" rather
    than silently dropping the field, which is loud enough to notice.
    """
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    s = str(v)
    if s.startswith("[") or s.startswith("{"):
        return s
    if "," in s and not (s.startswith('"') and s.endswith('"')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _quote_aware_split(text: str, sep: str = ",") -> List[str]:
    """Split on sep but treat double-quoted spans AND bracketed spans
    as opaque.

    Real-world A1111 params lines contain values like
    `Civitai resources: [{"modelName": "X", "weight": 1.0}]` where the
    commas inside the JSON-like structure are NOT separators. We track
    nesting depth for `[`/`]` and `{`/`}` (in addition to `"..."`) and
    only split on `sep` when we're at depth zero outside any quote.

    Escaped quotes (\\") inside a quoted span are preserved as part of
    the span.
    """
    parts: List[str] = []
    buf: List[str] = []
    in_quote = False
    bracket_depth = 0  # square brackets
    brace_depth = 0    # curly braces
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] == '"':
            buf.append(text[i:i + 2])
            i += 2
            continue
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
        elif not in_quote and ch == "[":
            bracket_depth += 1
            buf.append(ch)
        elif not in_quote and ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            buf.append(ch)
        elif not in_quote and ch == "{":
            brace_depth += 1
            buf.append(ch)
        elif not in_quote and ch == "}":
            brace_depth = max(0, brace_depth - 1)
            buf.append(ch)
        elif (
            ch == sep and not in_quote
            and bracket_depth == 0 and brace_depth == 0
        ):
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def format_a1111_params(meta: Dict[str, Any]) -> str:
    """Render a meta dict to the standard A1111 'parameters' string.

    Output shape:
        <prompt>
        Negative prompt: <negative>
        Steps: 30, Sampler: Euler a, CFG scale: 7, Seed: 12345, ...

    Returns '' when there's nothing useful to write (no prompt, no
    params).
    """
    canon = _normalize_meta(meta)
    if not canon:
        return ""

    lines: List[str] = []

    prompt = canon.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        lines.append(prompt.strip())

    neg = canon.get("Negative prompt")
    if isinstance(neg, str) and neg.strip():
        lines.append(f"Negative prompt: {neg.strip()}")

    param_pairs: List[str] = []
    seen: set[str] = set()
    for key in _A1111_STANDARD_ORDER:
        if key in canon and key not in _PARAMS_SKIP_KEYS:
            v = canon[key]
            param_pairs.append(f"{key}: {_format_param_value(v)}")
            seen.add(key)
    for key, value in canon.items():
        if key in seen or key in _PARAMS_SKIP_KEYS:
            continue
        param_pairs.append(f"{key}: {_format_param_value(value)}")
        seen.add(key)

    if param_pairs:
        lines.append(", ".join(param_pairs))

    # An A1111 string is empty if there's literally nothing to record;
    # don't write a chunk just to embed whitespace.
    return "\n".join(lines).strip()


def parse_a1111_params(text: str) -> Dict[str, Any]:
    """Parse an A1111 'parameters' string back into a canonical dict.

    Mirror of format_a1111_params: keys returned use Title-Case form
    (Steps, Sampler, CFG scale, Seed, Size, Model, Model hash, ...)
    plus 'prompt' and 'Negative prompt' for the prose fields.

    The parser is forgiving — it accepts multi-line prompts, an absent
    negative prompt, an absent params line, and quoted values
    containing commas. Anything truly unparseable produces a partial
    dict rather than raising, because giving up entirely would mean
    overwriting potentially-fine file data.
    """
    if not text or not text.strip():
        return {}

    text = text.replace("\r\n", "\n").strip()
    lines = text.split("\n")

    # Anchor: the params line is the LAST line that looks like a
    # key:value, key:value, ... list. We scan backwards because A1111
    # always puts the params line at the bottom, and prompts can
    # legitimately contain text like "Steps: how to use them" that
    # would falsely match a forward scan.
    def _is_params_line(line: str) -> bool:
        # Strong anchor: "Steps: " followed by a digit is what A1111
        # always emits at the start of its params line.
        if re.match(r"^Steps:\s+\d", line):
            return True
        # Weaker fallback for files where Steps was stripped or
        # renamed: at least two `Key: value` pairs separated by a
        # comma. Requires a Title-Case key to avoid matching prose.
        if re.match(r"^[A-Z][A-Za-z0-9 _\-]*:\s", line):
            return line.count(": ") >= 2 and "," in line
        return False

    params_idx: Optional[int] = None
    for i in range(len(lines) - 1, -1, -1):
        if _is_params_line(lines[i]):
            params_idx = i
            break

    result: Dict[str, Any] = {}

    # Find the negative prompt start, if any. A1111 always puts it
    # between prompt and params.
    neg_idx: Optional[int] = None
    scan_end = params_idx if params_idx is not None else len(lines)
    for i in range(scan_end):
        if lines[i].startswith("Negative prompt: "):
            neg_idx = i
            break

    # Slice out prompt / negative / params sections.
    if neg_idx is not None:
        prompt_lines = lines[:neg_idx]
        neg_lines = (
            [lines[neg_idx][len("Negative prompt: "):]]
            + lines[neg_idx + 1:scan_end]
        )
    else:
        prompt_lines = lines[:scan_end]
        neg_lines = []

    prompt = "\n".join(prompt_lines).strip()
    neg = "\n".join(neg_lines).strip()
    if prompt:
        result["prompt"] = prompt
    if neg:
        result["Negative prompt"] = neg

    if params_idx is not None:
        params_text = " ".join(lines[params_idx:])
        for part in _quote_aware_split(params_text, ","):
            if ":" not in part:
                continue
            k, _, v = part.partition(":")
            k = k.strip()
            v = v.strip()
            # Unwrap quoted values, restoring any escaped quotes.
            if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                v = v[1:-1].replace('\\"', '"')
            if k:
                result[k] = v

    return result


class MergeResult(NamedTuple):
    """Outcome of a non-destructive metadata merge.

    - `merged` — the union dict that will be re-formatted into the
      file's parameters string.
    - `added_keys` — keys the API contributed that the file didn't
      already have. Drives the dry-run report.
    - `changed` — convenience bool: True iff any rewrite is needed.
    - `notes` — human-readable list of special operations the merge
      performed beyond a plain gap-fill (JSON-prompt rescue, Lora
      hashes synthesis). Surfaced in the report so the user can see
      every non-trivial decision the merger made.
    """
    merged: Dict[str, Any]
    added_keys: List[str]
    changed: bool
    notes: List[str]


def _rescued_json_preserved(
    roundtrip: Dict[str, Any], original_prompt: Any
) -> bool:
    """True if the JSON-shaped original prompt is preserved under
    `Civitai source prompt` in the round-tripped dict.

    Compares as JSON-semantic-equal (parsed structures, not raw
    strings) because the A1111 params line flattens internal
    whitespace inside JSON values (newlines → spaces during the
    multi-line join). That whitespace shift is cosmetic, not data
    loss — both sides decode to the same object — so the safety
    net would otherwise reject SwarmUI / Civitai JSON prompts
    purely on indentation differences. Falls back to byte-identical
    string compare for cases where one side fails to parse.
    """
    csp = roundtrip.get("Civitai source prompt")
    if csp is None:
        return False
    csp_s = str(csp).strip()
    orig_s = str(original_prompt).strip()
    if csp_s == orig_s:
        return True
    try:
        return json.loads(csp_s) == json.loads(orig_s)
    except (json.JSONDecodeError, ValueError):
        return False


def _looks_like_json_dump(s: Any) -> bool:
    """True if `s` is a string whose first non-whitespace character
    starts a JSON object/array AND it actually parses as JSON.

    Real text prompts effectively never start with `{` or `[` and
    will almost never parse as JSON, so this is a strong signal
    that the field is misfiled workflow data rather than a prompt.
    We require BOTH conditions so a prompt that happens to start
    with `[masterpiece, ...]` (rare but possible) doesn't trigger.
    """
    if not isinstance(s, str):
        return False
    stripped = s.lstrip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _synthesize_lora_hashes(meta: Dict[str, Any]) -> Optional[str]:
    """Build the A1111-standard `Lora hashes` value from a merged
    meta dict, or return None when there's nothing to synthesize.

    The standard format is a comma-separated `name: hash` list:
        Lora hashes: "name1: hash1, name2: hash2"

    Two data sources are tried in order:
      1. `Civitai resources` — older Civitai PNG/API format,
         `[{name, hash, type}]`. Walks the list for type=="lora"
         entries with both name and hash present.
      2. `hashes` — Civitai API dict like
         `{"model": "...", "lora:Some LoRA": "abc123"}`. We pull
         keys starting with `lora:` and use the trailing portion
         as the LoRA name.

    Civitai's `civitaiResources` format has weight+modelVersionId
    but no hash, so it can't be used here — that information is
    still preserved in the `Civitai resources` JSON value verbatim.
    """
    pairs: List[Tuple[str, str]] = []
    seen: set[str] = set()

    res = meta.get("Civitai resources")
    if isinstance(res, str):
        # We may be looking at a value that came through the parser
        # as a JSON-encoded string. Try to inflate; if it doesn't
        # parse, just skip — Lora hashes is opportunistic.
        try:
            res = json.loads(res)
        except (json.JSONDecodeError, ValueError):
            res = None
    if isinstance(res, list):
        for r in res:
            if not isinstance(r, dict):
                continue
            if r.get("type") != "lora":
                continue
            name = r.get("name")
            h = r.get("hash")
            if name and h and name not in seen:
                pairs.append((str(name), str(h)))
                seen.add(name)

    hashes = meta.get("hashes")
    if isinstance(hashes, str):
        try:
            hashes = json.loads(hashes)
        except (json.JSONDecodeError, ValueError):
            hashes = None
    if isinstance(hashes, dict):
        for k, v in hashes.items():
            if not isinstance(k, str) or not k.startswith("lora:"):
                continue
            name = k[len("lora:"):]
            if name and v and name not in seen:
                pairs.append((name, str(v)))
                seen.add(name)

    if not pairs:
        return None
    return ", ".join(f"{n}: {h}" for n, h in pairs)


def merge_meta_nondestructive(
    file_meta: Dict[str, Any],
    api_meta: Dict[str, Any]
) -> MergeResult:
    """Merge api_meta into file_meta without overwriting any field
    that already has a non-empty value in file_meta.

    Core contract: file data is ground truth. API values fill only
    the keys the file is missing (or has stored as an empty string).

    Two policy exceptions extend this contract:

      1. JSON-prompt rescue. Some Civitai on-site generations file
         their workflow JSON into the `prompt` field, leaving the
         actual readable prompt unstored. When the FILE's prompt
         parses as JSON and the API has a real prompt, we move the
         JSON to `Civitai source prompt` (preserving it for the
         user's reference) and let the API's prompt take the
         `prompt` slot. The rescue is reported in `notes` so the
         operation is never silent.

      2. Lora hashes synthesis. A1111-style readers key off the
         `Lora hashes` field for LoRA reproducibility. When the
         file is missing it but the merged data has Civitai
         resources / hashes containing LoRA hashes, we synthesize
         the standard `name: hash, ...` value.
    """
    file_canon = _normalize_meta(file_meta)
    api_canon = _normalize_meta(api_meta)

    notes: List[str] = []

    # --- JSON-prompt rescue (only override of "file wins") ---
    file_prompt = file_canon.get("prompt")
    api_prompt = api_canon.get("prompt")
    if (
        _looks_like_json_dump(file_prompt)
        and isinstance(api_prompt, str)
        and api_prompt.strip()
        and not _looks_like_json_dump(api_prompt)
    ):
        # Stash the JSON content in a dedicated side field so the
        # original data is preserved verbatim and discoverable — and
        # so the round-trip safety check can confirm it wasn't lost.
        # Refuse to clobber an existing side field (extremely
        # unlikely but cheap to guard against).
        if "Civitai source prompt" not in file_canon:
            file_canon["Civitai source prompt"] = file_prompt
        # Remove the JSON from the prompt slot so the gap-fill loop
        # below treats prompt as missing and pulls from the API.
        del file_canon["prompt"]
        notes.append(
            "rescued JSON-shaped prompt → moved to "
            "'Civitai source prompt'; API prompt used instead"
        )

    # --- Standard non-destructive gap fill ---
    merged = dict(file_canon)
    added: List[str] = []
    for k, v in api_canon.items():
        existing = merged.get(k)
        if existing is None or (
            isinstance(existing, str) and not existing.strip()
        ):
            merged[k] = v
            added.append(k)

    # `Civitai source prompt` is technically a file-side addition,
    # not an API contribution, but the dry-run report should still
    # show it as a change so the user sees what happened.
    if (
        "Civitai source prompt" in merged
        and "Civitai source prompt" not in added
        and "Civitai source prompt" not in file_meta
    ):
        added.append("Civitai source prompt")

    # --- Lora hashes synthesis ---
    if "Lora hashes" not in merged:
        synth = _synthesize_lora_hashes(merged)
        if synth:
            merged["Lora hashes"] = synth
            added.append("Lora hashes")
            notes.append(
                "synthesized A1111-standard 'Lora hashes' from "
                "Civitai resources / hashes data"
            )

    return MergeResult(
        merged=merged,
        added_keys=added,
        changed=bool(added),
        notes=notes,
    )


# --- Low-level format detection / readers ---------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _detect_image_format(head: bytes) -> Optional[str]:
    """Return 'png', 'jpeg', 'webp', or None — based on magic bytes.

    Extension-based detection is unreliable for CivitAI downloads (we
    already sniff during download for that reason), so the embed path
    re-sniffs here rather than trusting `path.suffix`.
    """
    if head.startswith(_PNG_SIGNATURE):
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def _iter_png_chunks(data: bytes):
    """Yield (offset, length, type_bytes, data, crc_bytes) for each
    chunk in a PNG byte stream. Caller is responsible for the leading
    8-byte signature being present.
    """
    pos = len(_PNG_SIGNATURE)
    while pos < len(data):
        if pos + 8 > len(data):
            return
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]
        crc = data[pos + 8 + length:pos + 12 + length]
        yield pos, length, ctype, chunk_data, crc
        pos += 12 + length


def _build_text_chunk(keyword: str, text: str) -> bytes:
    """Build a PNG text chunk (tEXt or iTXt) for `keyword=text`.

    Picks the right chunk type for the content:
      - tEXt with ISO-8859-1 bytes when the text fits in latin-1.
        Matches the PNG spec, which says tEXt MUST be latin-1, and is
        what A1111 / Civitai / Forge produce for ASCII prompts.
      - iTXt with UTF-8 bytes (uncompressed) when the text contains
        any non-latin1 character. This is the spec-defined unicode
        carrier in PNG and is what Pillow's PngInfo.add_text falls
        back to for unicode strings — it's universally readable by
        downstream SD tooling.

    Using tEXt with UTF-8 bytes (an older non-conformant convention)
    would round-trip wrong through PIL: PIL decodes tEXt as latin-1,
    producing mojibake for unicode prompts on read. We avoid that
    failure mode entirely by choosing the right chunk type up front.
    """
    kw_bytes = keyword.encode("latin-1", errors="replace")
    try:
        text_bytes = text.encode("latin-1")
        chunk_data = kw_bytes + b"\x00" + text_bytes
        ctype = b"tEXt"
    except UnicodeEncodeError:
        # iTXt layout per the PNG spec:
        #   keyword \0 comp_flag(1) comp_method(1) lang \0
        #   translated_keyword \0 text
        # We emit uncompressed (comp_flag=0) with empty language and
        # translated-keyword fields; readers handle this consistently.
        text_bytes = text.encode("utf-8")
        chunk_data = (
            kw_bytes
            + b"\x00"        # keyword terminator
            + b"\x00"        # compression flag (0 = uncompressed)
            + b"\x00"        # compression method (ignored when flag=0)
            + b"\x00"        # language tag terminator (empty)
            + b"\x00"        # translated keyword terminator (empty)
            + text_bytes
        )
        ctype = b"iTXt"
    crc = zlib.crc32(ctype + chunk_data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(chunk_data))
        + ctype
        + chunk_data
        + struct.pack(">I", crc)
    )


def _decode_text_bytes(text_bytes: bytes) -> str:
    """Decode PNG text-chunk bytes the way the SD ecosystem expects.

    tEXt per spec is ISO-8859-1; iTXt is UTF-8. But older A1111
    versions wrote tEXt with raw UTF-8 bytes (non-spec). To handle
    both cleanly without mojibake either way, we try strict UTF-8
    first — that covers spec-compliant iTXt, ASCII tEXt, and the
    legacy A1111 utf8-in-tEXt convention. We fall back to latin-1
    only when UTF-8 fails, which is the spec-compliant decoding for
    tEXt with accented characters.
    """
    try:
        return text_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return text_bytes.decode("latin-1", errors="replace")


def _read_png_parameters(path: Path) -> str:
    """Return the raw `parameters` text value from a PNG, or ''.

    Reads via raw chunk parsing rather than PIL so the caller doesn't
    accidentally re-encode anything when they later write. Handles
    both tEXt and iTXt chunks; for iTXt with compression we
    transparently inflate before decoding.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if not data.startswith(_PNG_SIGNATURE):
        return ""
    for _, _, ctype, cdata, _ in _iter_png_chunks(data):
        if ctype == b"tEXt":
            sep = cdata.find(b"\x00")
            if sep == -1:
                continue
            keyword = cdata[:sep].decode("latin-1", errors="replace")
            if keyword == "parameters":
                return _decode_text_bytes(cdata[sep + 1:])
        elif ctype == b"iTXt":
            sep = cdata.find(b"\x00")
            if sep == -1:
                continue
            keyword = cdata[:sep].decode("latin-1", errors="replace")
            if keyword != "parameters":
                continue
            # iTXt: keyword\0 comp_flag(1) comp_method(1) lang\0
            # translated_keyword\0 text
            rest = cdata[sep + 1:]
            if len(rest) < 2:
                continue
            comp_flag = rest[0]
            # Skip past the language tag and translated keyword.
            after_method = rest[2:]
            lang_end = after_method.find(b"\x00")
            if lang_end == -1:
                continue
            after_lang = after_method[lang_end + 1:]
            tkw_end = after_lang.find(b"\x00")
            if tkw_end == -1:
                continue
            text_bytes = after_lang[tkw_end + 1:]
            if comp_flag == 1:
                try:
                    text_bytes = zlib.decompress(text_bytes)
                except zlib.error:
                    continue
            # iTXt is always UTF-8 per spec; replace on malformed
            # input rather than raising, since dropping the whole
            # chunk would be more destructive.
            return text_bytes.decode("utf-8", errors="replace")
    return ""


def _read_jpeg_parameters(path: Path) -> str:
    """Return the EXIF UserComment text from a JPEG, or ''.

    UserComment is encoded as `<charset 8 bytes><payload>` per the
    EXIF spec. piexif.helper.UserComment.load handles all four standard
    charsets (UNICODE/ASCII/JIS/undefined) and returns a Python string.
    """
    try:
        exif_dict = piexif.load(str(path))
    except Exception:
        return ""
    exif_ifd = exif_dict.get("Exif") or {}
    uc = exif_ifd.get(piexif.ExifIFD.UserComment)
    if not uc:
        return ""
    try:
        return piexif.helper.UserComment.load(uc) or ""
    except Exception:
        # Some files store raw text without the charset header — fall
        # back to a best-effort decode rather than dropping the value.
        if isinstance(uc, bytes):
            return uc.decode("utf-8", errors="replace").lstrip("\x00")
        return str(uc)


# --- Low-level writers ---------------------------------------------

class EmbedError(Exception):
    """Raised when an embed operation cannot be completed safely.

    The caller is expected to catch this, log it, and move on — one
    bad file never aborts a batch.
    """


def _atomic_replace(tmp: Path, target: Path) -> None:
    """Promote a fully-written tempfile to its final name atomically.

    os.replace is atomic on the same filesystem (POSIX rename(2)
    semantics), so a crash mid-rename either leaves the original or
    the new file — never a half-state.
    """
    os.replace(tmp, target)


def _maybe_backup(
    path: Path,
    enabled: bool,
    backup_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Copy the original to a `.civitai-orig` backup once, if backups
    are on. Subsequent calls don't overwrite an existing backup — that
    would lose the true original after a second apply.

    Default layout (backup_dir=None) places the backup as a sibling
    next to the original: `<name>.civitai-orig`.

    When backup_dir is set, the backup goes to
    `<backup_dir>/<parent_folder_name>/<name>.civitai-orig` — keeps
    the source folder clean and groups backups by model folder.
    """
    if not enabled:
        return None
    if backup_dir is not None:
        sub = backup_dir / path.parent.name
        sub.mkdir(parents=True, exist_ok=True)
        bk = sub / (path.name + ".civitai-orig")
    else:
        bk = path.with_name(path.name + ".civitai-orig")
    if not bk.exists():
        shutil.copy2(path, bk)
    return bk


def _verify_png_after_write(
    tmp: Path,
    expected_size: Tuple[int, int],
    expected_params: str,
) -> None:
    """Open the tempfile and confirm it's a valid PNG with the params
    we just wrote and dimensions matching the original. Raises
    EmbedError if anything is off.
    """
    try:
        with Image.open(tmp) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise EmbedError(f"verify failed (not a valid PNG): {exc}")
    # Image.verify closes the image; reopen to read size + chunks.
    try:
        with Image.open(tmp) as img:
            if img.format != "PNG":
                raise EmbedError(
                    f"format changed: PNG → {img.format}"
                )
            if img.size != expected_size:
                raise EmbedError(
                    f"dimensions changed: {expected_size} → {img.size}"
                )
    except (UnidentifiedImageError, OSError) as exc:
        raise EmbedError(f"reopen failed: {exc}")
    actual = _read_png_parameters(tmp)
    if actual != expected_params:
        raise EmbedError(
            "round-trip mismatch: written parameters not readable back"
        )


def _verify_jpeg_after_write(
    tmp: Path,
    expected_size: Tuple[int, int],
    expected_params: str,
) -> None:
    """Same shape as _verify_png_after_write but for JPEG output."""
    try:
        with Image.open(tmp) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise EmbedError(f"verify failed (not a valid JPEG): {exc}")
    try:
        with Image.open(tmp) as img:
            if img.format != "JPEG":
                raise EmbedError(
                    f"format changed: JPEG → {img.format}"
                )
            if img.size != expected_size:
                raise EmbedError(
                    f"dimensions changed: {expected_size} → {img.size}"
                )
    except (UnidentifiedImageError, OSError) as exc:
        raise EmbedError(f"reopen failed: {exc}")
    actual = _read_jpeg_parameters(tmp)
    if actual != expected_params:
        raise EmbedError(
            "round-trip mismatch: written UserComment not readable back"
        )
    # Strict-EXIF compliance: refuse to ship a file whose sub-IFD
    # layout would trip a spec-strict reader (kamadak-exif, exiv2
    # strict). Catches any future regression in the piexif-dump
    # monkey-patch before it can land on disk.
    compliance_issues = validate_subifd_compliance(tmp.read_bytes())
    if compliance_issues:
        raise EmbedError(
            "strict-EXIF compliance check failed: "
            + "; ".join(compliance_issues)
        )


def _write_png_with_parameters(path: Path, params_string: str) -> None:
    """Rewrite a PNG, replacing (or inserting) a single `parameters`
    tEXt chunk while leaving every other byte untouched.

    Implementation: walk chunks, drop any pre-existing `parameters`
    tEXt/iTXt, insert the new tEXt right before the first IDAT.
    Recomputes CRC for the new chunk only. The IDAT (pixel) stream is
    never decoded.
    """
    data = path.read_bytes()
    if not data.startswith(_PNG_SIGNATURE):
        raise EmbedError("not a PNG (signature missing)")

    new_chunk = _build_text_chunk("parameters", params_string)

    out = bytearray(_PNG_SIGNATURE)
    inserted = False
    for _, _, ctype, cdata, crc in _iter_png_chunks(data):
        # Drop existing parameters chunks (tEXt or iTXt) — we replace
        # them with our merged version. Other tEXt/iTXt chunks pass
        # through unchanged.
        if ctype in (b"tEXt", b"iTXt"):
            sep = cdata.find(b"\x00")
            if sep != -1:
                keyword = cdata[:sep].decode(
                    "latin-1", errors="replace"
                )
                if keyword == "parameters":
                    continue
        if not inserted and ctype == b"IDAT":
            out.extend(new_chunk)
            inserted = True
        # Re-emit the chunk byte-for-byte (length+type+data+crc).
        out.extend(struct.pack(">I", len(cdata)))
        out.extend(ctype)
        out.extend(cdata)
        out.extend(crc)

    if not inserted:
        # No IDAT means a malformed PNG — refuse rather than guess.
        raise EmbedError("no IDAT chunk found; refusing to write")

    tmp = path.with_name(path.name + ".tmp.civitai-meta")
    try:
        with open(tmp, "wb") as f:
            f.write(bytes(out))
            f.flush()
            os.fsync(f.fileno())
        with Image.open(path) as orig:
            expected_size = orig.size
        _verify_png_after_write(tmp, expected_size, params_string)
        _atomic_replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _write_jpeg_with_parameters(
    path: Path, params_string: str
) -> None:
    """Embed UserComment via piexif. Only the EXIF APP1 segment of the
    JPEG is rewritten; the compressed image stream is untouched
    byte-for-byte.

    Existing EXIF fields (camera tags, datetimes, etc.) are preserved
    by loading-modifying-dumping rather than building a fresh EXIF
    dict from scratch.
    """
    try:
        existing = piexif.load(str(path))
    except Exception:
        # File has no EXIF — start from an empty dict, which piexif
        # accepts and renders into a minimal valid APP1 segment.
        existing = {
            "0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None
        }

    existing.setdefault("Exif", {})
    existing["Exif"][piexif.ExifIFD.UserComment] = (
        piexif.helper.UserComment.dump(params_string, encoding="unicode")
    )

    # piexif raises on certain malformed tags it can't round-trip
    # (sometimes thumbnail data or oversized values). When that
    # happens, retry without the thumbnail — it's the most common
    # culprit and is non-essential.
    try:
        exif_bytes = piexif.dump(existing)
    except Exception:
        existing["thumbnail"] = None
        existing["1st"] = {}
        exif_bytes = piexif.dump(existing)

    tmp = path.with_name(path.name + ".tmp.civitai-meta")
    try:
        # Splice the new EXIF APP1 into a byte-perfect copy of the
        # original. Unlike piexif.insert (which drops APP0 JFIF on
        # most JPEGs), splice_exif_app1 preserves every other segment
        # verbatim — compressed scan data included. The original file
        # is read; the tempfile is written from scratch.
        original_bytes = path.read_bytes()
        new_bytes = splice_exif_app1(original_bytes, exif_bytes)
        with open(tmp, "wb") as f:
            f.write(new_bytes)
        with Image.open(path) as orig:
            expected_size = orig.size
        _verify_jpeg_after_write(tmp, expected_size, params_string)
        _atomic_replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# --- Top-level orchestrator ----------------------------------------

class EnrichStatus:
    """Discrete outcomes from enrich_file. Strings so the report can
    bucket without translating an enum."""
    CHANGED = "changed"
    NO_CHANGE = "no_change"
    NO_API_DATA = "no_api_data"
    UNSUPPORTED = "unsupported_format"
    NOT_FOUND = "not_found"
    ERROR = "error"


class EnrichResult(NamedTuple):
    """Per-file outcome of an enrichment call.

    - `status` — one of the EnrichStatus constants.
    - `added_keys` — keys the merge contributed when status==CHANGED.
    - `notes` — human-readable lines describing any non-trivial
      decisions the merge made (e.g. "rescued JSON-shaped prompt").
      Empty for plain gap-fill operations. Surfaced in the report.
    - `error` — populated only when status==ERROR.
    """
    status: str
    added_keys: List[str]
    notes: List[str]
    error: Optional[str]


def enrich_file(
    path: Path,
    api_meta: Optional[Dict[str, Any]],
    *,
    dry_run: bool = False,
    backup: bool = False,
    backup_dir: Optional[Path] = None,
) -> EnrichResult:
    """Merge api_meta into the file at `path`, non-destructively.

    - Existing embedded keys are NEVER overwritten.
    - A single `parameters` slot is rewritten (no duplicates).
    - If the merge produces no new keys, the file is not touched.
    - WEBP and other formats we can't safely patch return UNSUPPORTED
      rather than raising.

    Set dry_run=True to compute what would change without writing.
    Set backup=True to copy the original to `<name>.civitai-orig`
    before the first write (subsequent writes do not re-backup).
    """
    if not path.exists():
        return EnrichResult(EnrichStatus.NOT_FOUND, [], [], None)

    if not api_meta:
        return EnrichResult(EnrichStatus.NO_API_DATA, [], [], None)

    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError as exc:
        return EnrichResult(
            EnrichStatus.ERROR, [], [], f"read failed: {exc}"
        )

    fmt = _detect_image_format(head)
    if fmt not in ("png", "jpeg"):
        return EnrichResult(EnrichStatus.UNSUPPORTED, [], [], None)

    try:
        existing_raw = (
            _read_png_parameters(path)
            if fmt == "png"
            else _read_jpeg_parameters(path)
        )
    except Exception as exc:
        return EnrichResult(
            EnrichStatus.ERROR, [], [], f"read existing failed: {exc}"
        )

    file_meta = parse_a1111_params(existing_raw) if existing_raw else {}
    merge = merge_meta_nondestructive(file_meta, api_meta)
    if not merge.changed:
        return EnrichResult(EnrichStatus.NO_CHANGE, [], [], None)

    new_params = format_a1111_params(merge.merged)
    if not new_params:
        # API meta normalised to nothing (e.g. all empty strings).
        return EnrichResult(EnrichStatus.NO_CHANGE, [], [], None)

    # Belt-and-braces: if the formatted output is byte-identical to
    # what's already in the file, skip the write — protects against
    # over-eager "added_keys" counts if normalisation round-trips
    # unchanged.
    if new_params == existing_raw:
        return EnrichResult(EnrichStatus.NO_CHANGE, [], [], None)

    # Round-trip safety: parse what we're about to write and confirm
    # every key/value the file currently has is preserved (under
    # SOME key in the output). If our formatter would drop or mangle
    # an existing field, REFUSE to write — better to leave the file
    # alone than silently lose something the user wanted to keep.
    #
    # The JSON-prompt rescue is the one intentional value change:
    # the file's JSON `prompt` is moved to `Civitai source prompt`
    # so the API's real prompt can take the prompt slot. We allow
    # that specific case by checking the rescued JSON is still
    # present under the side key.
    #
    # We iterate the NORMALIZED file_meta (canonical Title-Case keys)
    # so that Civitai-style aliases like `cfgScale` don't appear
    # missing from the round-trip — the merger already renamed them
    # to their A1111 equivalent (`CFG scale`), and the safety net
    # should follow the same convention. Empty values and synthesized
    # composites (Size from width/height) are handled identically on
    # both sides because the round-trip dict came from formatted
    # canonical output.
    if file_meta:
        roundtrip = parse_a1111_params(new_params)
        file_meta_canon = _normalize_meta(file_meta)
        for k, v in file_meta_canon.items():
            # Empty/whitespace-only file values are intentionally
            # dropped by _normalize_meta — they carry no information.
            # Their absence from the round-trip output is expected,
            # not data loss. (E.g. a ComfyUI export that left
            # `Model: ` with no value.)
            if isinstance(v, str) and not v.strip():
                continue
            if k not in roundtrip:
                # Permit the JSON-prompt rescue: original prompt
                # value is preserved under `Civitai source prompt`.
                if (k == "prompt" and _looks_like_json_dump(v)
                        and _rescued_json_preserved(roundtrip, v)):
                    continue
                return EnrichResult(
                    EnrichStatus.ERROR,
                    [],
                    [],
                    f"round-trip would drop key {k!r}; refusing to "
                    f"write to protect existing data",
                )
            # Compare as strings — types may shift (int vs str)
            # across the format/parse boundary, but the textual
            # content must match.
            if str(roundtrip[k]).strip() != str(v).strip():
                # Same rescue permission: prompt value changed
                # because we moved the JSON; check it's preserved.
                if (k == "prompt" and _looks_like_json_dump(v)
                        and _rescued_json_preserved(roundtrip, v)):
                    continue
                return EnrichResult(
                    EnrichStatus.ERROR,
                    [],
                    [],
                    f"round-trip would change value of {k!r} "
                    f"({v!r} → {roundtrip[k]!r}); refusing to write",
                )

    if dry_run:
        return EnrichResult(
            EnrichStatus.CHANGED,
            list(merge.added_keys),
            list(merge.notes),
            None,
        )

    try:
        _maybe_backup(path, backup, backup_dir)
        if fmt == "png":
            _write_png_with_parameters(path, new_params)
        else:
            _write_jpeg_with_parameters(path, new_params)
    except Exception as exc:
        return EnrichResult(EnrichStatus.ERROR, [], [], str(exc))

    return EnrichResult(
        EnrichStatus.CHANGED,
        list(merge.added_keys),
        list(merge.notes),
        None,
    )


# ---------------------------------------------------------------------
# End of metadata embedding helpers
# ---------------------------------------------------------------------


class ImageFetchResult(NamedTuple):
    """Outcome of a paginated image fetch.

    Fields:

    - `images`: items that passed every filter the caller supplied
      via `accept_fn`. Length is at most `limit`.

    - `pages_walked`: total API requests made across every version
      queried — one per page. A pure single-call fetch reports 1.

    - `candidates_seen`: count of image-typed items the API
      returned across all pages (after the internal video skip).
      Useful for "we walked the whole gallery and only found 3
      matches" diagnostics.

    - `drops_by_reason`: maps each reason returned by `accept_fn`
      (e.g. "known", "no_meta", "begging") to its count. The
      built-in video skip contributes under the `"video"` key.
      `candidates_seen` == `len(images) + sum(non-video drops)`.

    - `hit_page_limit`: True if pagination stopped at
      `max_pages_per_version` for any version before exhausting
      the cursor. Lets callers decide whether to raise the cap or
      accept that the gallery genuinely doesn't contain enough
      matches.
    """
    images: List[Dict[str, Any]]
    pages_walked: int
    candidates_seen: int
    drops_by_reason: Dict[str, int]
    hit_page_limit: bool


class CivitAIFetcher:
    """Handles fetching data from CivitAI API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://civitai.com/api/v1",
        max_retries: int = 3,
        backoff_factor: int = 1
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_session(max_retries, backoff_factor)

    def _create_session(
        self,
        max_retries: int,
        backoff_factor: int
    ) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if self.api_key:
            session.headers.update(
                {"Authorization": f"Bearer {self.api_key}"}
            )

        return session

    @staticmethod
    def extract_model_id(
        url_or_id: str
    ) -> tuple[int, Optional[int]]:
        """Extract model ID and optional modelVersionId from URL or ID

        Pure function — does not touch the session or API key, so it's
        a staticmethod. Useful for cheap input validation (e.g. the
        interactive prompt validator) without paying to construct a
        requests session and retry adapter.

        Returns:
            tuple: (model_id, model_version_id)
        """
        model_version_id = None

        # If it's just a number, return it
        if url_or_id.isdigit():
            return int(url_or_id), None

        # Try to extract modelVersionId from query parameters
        version_match = re.search(r'modelVersionId=(\d+)', url_or_id)
        if version_match:
            model_version_id = int(version_match.group(1))

        # Try to extract model ID from URL
        patterns = [
            r'civitai\.com/models/(\d+)',
            r'civitai\.com/api/v1/models/(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return int(match.group(1)), model_version_id

        raise ValueError(f"Could not extract model ID from: {url_or_id}")

    def get_model_details(self, model_id: int) -> Dict[str, Any]:
        """Fetch model details from CivitAI API"""
        url = f"{self.base_url}/models/{model_id}"
        print(f"Fetching model details from: {url}")

        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    # Magic byte signatures for supported image formats.
    # Videos and other formats are intentionally excluded — we filter
    # them out at the API level and as a defense in depth on download.
    _IMAGE_SIGNATURES = (
        (b'\xff\xd8\xff', 'jpeg'),
        (b'\x89PNG\r\n\x1a\n', 'png'),
        (b'GIF87a', 'gif'),
        (b'GIF89a', 'gif'),
    )

    @staticmethod
    def detect_image_extension(head: bytes) -> Optional[str]:
        """Detect image extension from file magic bytes.

        Returns the extension (without leading dot) for supported image
        formats, or None for unknown/unsupported formats including
        videos.
        """
        for sig, ext in CivitAIFetcher._IMAGE_SIGNATURES:
            if head.startswith(sig):
                return ext
        # WEBP: RIFF....WEBP
        if len(head) >= 12 and head[:4] == b'RIFF' and head[8:12] == b'WEBP':
            return 'webp'
        return None

    # CivitAI's /images endpoint caps a single request at 200
    # results. Always ask for the max — we paginate via cursor for
    # anything beyond, and a larger page size means fewer
    # round-trips and less time spent on api_delay sleeps.
    _IMAGES_API_PAGE_SIZE = 200

    def get_model_images(
        self,
        model_data: Dict[str, Any],
        limit: int = 200,
        sort: str = "Most Reactions",
        period: str = "AllTime",
        nsfw: Optional[bool] = None,
        specific_version_id: Optional[int] = None,
        api_delay: float = 1.5,
        accept_fn: Optional[
            Callable[[Dict[str, Any]], Optional[str]]
        ] = None,
        max_pages_per_version: int = 25,
    ) -> ImageFetchResult:
        """Fetch up to `limit` images for a model, paginating as needed.

        Walks `metadata.nextCursor` per the CivitAI API contract,
        always requesting the API maximum (200) per call. For each
        item returned, applies the built-in video filter and then
        the optional `accept_fn`:

        - `accept_fn(image) -> None`   → keep the image
        - `accept_fn(image) -> "<r>"`  → drop, count under reason `r`

        Pagination stops when any of these is true (in priority
        order):

        1. `limit` images have been accepted (target met)
        2. The current response has no `nextCursor` (gallery
           exhausted for this version)
        3. `max_pages_per_version` API calls have been made for
           this version (safety cap to bound runaway loops)

        With multiple versions and no `specific_version_id`, each
        version is walked in turn. The cursor is per-version — it
        does not carry across versions.

        Args:
            model_data: Model payload from /api/v1/models/:id
            limit: Target count of accepted images (post-filter).
                The function returns at most this many; it can
                return fewer if the gallery is exhausted or
                max_pages is hit.
            sort: One of Most Reactions, Most Comments,
                Most Collected, Newest, Oldest.
            period: One of AllTime, Year, Month, Week, Day.
            nsfw: True=NSFW-only, False=SFW-only, None=no filter.
            specific_version_id: When set, only that version's
                gallery is queried.
            api_delay: Sleep between successive API calls. Honored
                between pages and between versions.
            accept_fn: Optional per-image predicate. Returns None
                to keep, or a short reason string to drop. The
                reason is counted under drops_by_reason in the
                result.
            max_pages_per_version: Safety cap on pages walked per
                version. At the default 25 pages × 200 items, the
                fetcher will examine up to 5000 candidates per
                version before giving up — enough for any realistic
                gallery while still bounding runaway pagination.

        Returns:
            ImageFetchResult with the accepted images, page count,
            candidate count, per-reason drop counts (including
            "video" for the built-in filter), and a flag indicating
            whether the page cap was hit on any version.
        """
        accepted: List[Dict[str, Any]] = []
        drops: Dict[str, int] = {}
        candidates_seen = 0
        pages_walked = 0
        hit_page_limit = False

        versions = model_data.get("modelVersions", [])
        if not versions:
            print("No model versions found")
            return ImageFetchResult(
                accepted, pages_walked, candidates_seen,
                drops, hit_page_limit
            )

        if specific_version_id:
            versions = [
                v for v in versions if v.get("id") == specific_version_id
            ]
            if not versions:
                print(
                    f"⚠️  Warning: Model version ID {specific_version_id} "
                    "not found in this model"
                )
                return ImageFetchResult(
                    accepted, pages_walked, candidates_seen,
                    drops, hit_page_limit
                )
            print(
                f"🎯 Filtering to specific version ID: "
                f"{specific_version_id}"
            )

        url = f"{self.base_url}/images"

        for version in versions:
            if len(accepted) >= limit:
                break

            version_id = version.get("id")
            version_name = version.get("name", "Unknown")
            print(
                f"Fetching images for version: {version_name} "
                f"(ID: {version_id})..."
            )

            cursor: Optional[Any] = None
            pages_this_version = 0
            error_on_this_version = False

            while len(accepted) < limit:
                if pages_this_version >= max_pages_per_version:
                    hit_page_limit = True
                    print(
                        f"  ⚠️  Reached the page cap "
                        f"({max_pages_per_version}) for this version "
                        f"without filling the target. Raise "
                        f"max_pages_per_version if you need to walk "
                        "deeper into the gallery."
                    )
                    break

                params: Dict[str, Any] = {
                    "modelVersionId": version_id,
                    "limit": self._IMAGES_API_PAGE_SIZE,
                    "sort": sort,
                    "period": period,
                }
                if nsfw is not None:
                    params["nsfw"] = str(nsfw).lower()
                if cursor is not None:
                    params["cursor"] = cursor

                try:
                    response = self.session.get(
                        url, params=params, timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    print(
                        f"  Error fetching images for version "
                        f"{version_id}: {e}"
                    )
                    error_on_this_version = True
                    break

                items = data.get("items") or []
                metadata = data.get("metadata") or {}

                pages_walked += 1
                pages_this_version += 1

                page_kept = 0
                page_videos = 0
                page_drops: Dict[str, int] = {}

                for item in items:
                    if item.get("type", "image") != "image":
                        page_videos += 1
                        drops["video"] = drops.get("video", 0) + 1
                        continue
                    candidates_seen += 1
                    if accept_fn is None:
                        accepted.append(item)
                        page_kept += 1
                    else:
                        reason = accept_fn(item)
                        if reason is None:
                            accepted.append(item)
                            page_kept += 1
                        else:
                            drops[reason] = drops.get(reason, 0) + 1
                            page_drops[reason] = (
                                page_drops.get(reason, 0) + 1
                            )
                    if len(accepted) >= limit:
                        break

                # Per-page progress line. Show the running total
                # vs target so the user can see when we're getting
                # close to done. Skip on the trivial single-call
                # case (1 page, no drops) to keep output tight.
                detail_parts = []
                if page_drops:
                    drop_str = ', '.join(
                        f'{n} {r}' for r, n in sorted(page_drops.items())
                    )
                    detail_parts.append(f"dropped: {drop_str}")
                if page_videos:
                    detail_parts.append(f"{page_videos} video(s)")
                detail = (
                    f" ({'; '.join(detail_parts)})" if detail_parts else ""
                )
                print(
                    f"  page {pages_this_version}: "
                    f"got {len(items)} items, kept {page_kept}"
                    f"{detail} — accepted {len(accepted)}/{limit}"
                )

                # Stop conditions: empty page (shouldn't normally
                # happen) or no cursor for the next page (gallery
                # exhausted for this version).
                next_cursor = metadata.get("nextCursor")
                if not items or next_cursor in (None, "", 0):
                    break
                cursor = next_cursor

                # Rate limit only when we're going to make another
                # call. Skip the sleep on the final page so the
                # user doesn't wait pointlessly.
                if len(accepted) < limit:
                    time.sleep(api_delay)

            # Between versions, honor api_delay too — but only if
            # we actually plan to query another version.
            if (
                not error_on_this_version
                and len(accepted) < limit
                and version is not versions[-1]
            ):
                time.sleep(api_delay)

        return ImageFetchResult(
            images=accepted,
            pages_walked=pages_walked,
            candidates_seen=candidates_seen,
            drops_by_reason=drops,
            hit_page_limit=hit_page_limit,
        )

    def download_image(
        self,
        url: str,
        output_dir: Path,
        image_id: Any,
        api_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Download an image and save with extension inferred from bytes.

        The CivitAI CDN serves files whose URL extension doesn't always
        match the actual content (a URL ending in .jpeg may be a PNG),
        so we sniff the magic bytes and pick the extension ourselves.
        Returns the final saved Path, or None if the download failed or
        the content wasn't a supported image format (e.g. video).

        When `api_meta` is supplied, the saved file is also enriched
        with the generation parameters from the API. This is a no-op
        for files the CDN already served with full embedded metadata,
        and otherwise patches in the missing fields non-destructively.
        Enrichment failures don't fail the download — we log and keep
        the raw file, since a metadata-less image is still useful.
        """
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            content = response.content
            ext = self.detect_image_extension(content[:16])
            if ext is None:
                print(
                    f"  ⏭️  Skipping {image_id}: unsupported format "
                    f"(magic={content[:8].hex()})"
                )
                return None

            # CivitAI's CDN occasionally serves PNGs with a bad CRC
            # on the tEXt chunk holding generation parameters. The
            # pixel data is fine, but PIL refuses to open the file
            # — which breaks every downstream consumer (image library
            # apps, the backfill enrich path, etc.). Repair the CRC
            # in place before writing so the file we save is one PIL
            # / browsers / SagaSigil can actually open. The repair
            # never alters chunk content (only the 4-byte CRC field),
            # so embedded metadata stays byte-identical.
            if ext == "png":
                rr = repair_png_bytes(content)
                if rr.structural_error:
                    # Beyond simple CRC bit-rot. Save the original
                    # bytes verbatim — we won't make guesses on a
                    # malformed file.
                    print(
                        f"      ⚠️  PNG repair skipped for "
                        f"{image_id}: {rr.structural_error}"
                    )
                elif rr.notes:
                    content = rr.data
                    print(
                        f"      🔧 Repaired {len(rr.notes)} bad "
                        f"chunk CRC(s) in {image_id} (CDN defect)"
                    )

            # Atomic write: stage the bytes in a dotfile sibling,
            # fsync, then rename over the final path. This way a
            # Ctrl-C or network drop mid-write never leaves a
            # truncated `{id}.{ext}` on disk — the existence check
            # in find_image_filename would otherwise treat that
            # partial file as "already downloaded" and skip the
            # re-download on the next run, leaving a broken embed.
            # The leading `.` keeps the tmp file out of glob matches
            # in find_image_filename and find_notes_by_source.
            output_path = output_dir / f"{image_id}.{ext}"
            tmp_path = output_dir / f".{image_id}.{ext}.tmp"
            try:
                with open(tmp_path, 'wb') as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.replace(output_path)
            except Exception:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise

            if api_meta:
                result = enrich_file(output_path, api_meta)
                if result.status == EnrichStatus.ERROR:
                    # The original download is intact (enrich uses
                    # tempfile+atomic-rename); just warn so the user
                    # knows this image's metadata wasn't patched.
                    print(
                        f"      ⚠️  Metadata embed failed for "
                        f"{image_id}: {result.error}"
                    )
                elif result.status == EnrichStatus.CHANGED:
                    print(
                        f"      ➕ Embedded {len(result.added_keys)} "
                        f"missing field(s): "
                        f"{', '.join(result.added_keys[:5])}"
                        + ("..." if len(result.added_keys) > 5 else "")
                    )

            return output_path
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None


class ObsidianPageGenerator:
    """Generates Obsidian markdown pages"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        obs_config = config.get("obsidian", {})
        self.vault_path = Path(obs_config.get("vault_path", "."))
        media_folder = obs_config.get(
            "media_folder",
            "zzMedia/Model and Lora Example Images"
        )
        self.media_folder = self.vault_path / media_folder
        self.base_directory = obs_config.get(
            "base_directory",
            "1 - Saga of Gone/8 - Diffusion"
        )
        self.directories = obs_config.get("directories", {})
        self.type_directories = obs_config.get("type_directories", {})

    def get_note_directory(
        self,
        model_data: Dict[str, Any],
        specific_version_id: Optional[int] = None
    ) -> Path:
        """Get the directory path for saving the note"""
        base_model = self.detect_base_model(
            model_data,
            specific_version_id
        )
        model_type = model_data.get('type', 'Unknown').upper()

        # Default base directory
        base_dir = self.vault_path / self.base_directory

        # Determine base model subdirectory
        if base_model == "SDXL":
            subdir = self.directories.get("sdxl", "3 - SDXL")
        elif base_model == "SD15":
            subdir = self.directories.get("sd15", "2 - SD15")
        elif base_model == "FLUX.DEV":
            subdir = self.directories.get("flux", "1 - FLUX")
        elif base_model == "PONY":
            subdir = self.directories.get("pony", "4 - PONY")
        elif base_model == "ILLUST":
            subdir = self.directories.get("illust", "5 - ILLUST")
        else:
            # Fallback for unknown base models
            subdir = self.directories.get("other", "9 - Other")

        base_dir = base_dir / subdir

        # Add type subdirectory
        if model_type == 'LORA':
            type_subdir = self.type_directories.get("lora", "Lora")
        elif model_type == 'CHECKPOINT':
            type_subdir = self.type_directories.get(
                "checkpoint",
                "Models"
            )
        elif model_type == 'TEXTUALINVERSION':
            type_subdir = self.type_directories.get(
                "textualinversion",
                "Embeddings"
            )
        else:
            type_subdir = self.type_directories.get("other", "Other")

        return base_dir / type_subdir

    def get_clean_folder_name(self, model_name: str) -> str:
        """Get clean folder name without underscores"""
        return model_name.replace("_", " ")

    def sanitize_filename(self, name: str) -> str:
        """Sanitize filename for filesystem"""
        # Remove or replace invalid characters
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = re.sub(r'\s+', '_', name)
        return name

    @staticmethod
    def sanitize_folder_name(name: str) -> str:
        """Sanitize a folder name for cross-platform filesystems.

        Unlike sanitize_filename, this keeps spaces and parentheses
        intact because folder names are primarily for humans browsing
        the vault. Strips characters that are invalid on Windows/macOS
        and trims trailing dots/spaces (also a Windows constraint).
        """
        name = re.sub(r'[/\\:*?"<>|]', '_', name)
        name = re.sub(r'\s+', ' ', name).strip()
        name = name.rstrip('. ')
        return name or 'unnamed'

    @classmethod
    def build_image_folder_name(
        cls,
        model_data: Dict[str, Any],
        specific_version_id: Optional[int]
    ) -> str:
        """Construct the image folder name for this download.

        Format: `{model_name} ({version_name})` when a specific
        version was requested, else just `{model_name}`. Falls back
        to ID-based names when the API has no human name.
        """
        model_id = model_data.get('id')
        model_name = model_data.get('name') or f'[unnamed-{model_id}]'

        if specific_version_id is None:
            return cls.sanitize_folder_name(model_name)

        version = next(
            (
                v for v in model_data.get('modelVersions', [])
                if v.get('id') == specific_version_id
            ),
            None
        )
        version_name = (
            (version.get('name') if version else None)
            or f'v{specific_version_id}'
        )
        return cls.sanitize_folder_name(
            f'{model_name} ({version_name})'
        )

    def detect_base_model(
        self,
        model_data: Dict[str, Any],
        specific_version_id: Optional[int] = None
    ) -> Optional[str]:
        """Detect base model from version name or tags

        Args:
            model_data: Model data from API
            specific_version_id: If provided, detect base model from
                                this specific version
        """
        # Check model versions for base model indicators
        versions = model_data.get("modelVersions", [])
        if not versions:
            return None

        # If specific version ID provided, use that version for detection
        if specific_version_id:
            target_version = next(
                (v for v in versions if v.get("id") == specific_version_id),
                None
            )
            if target_version:
                versions = [target_version]

        # Use first version (or the specific version if filtered above)
        version_name = versions[0].get("name", "").upper()
        base_model = versions[0].get("baseModel", "").upper()

        # Check baseModel field first
        if "FLUX" in base_model:
            return "FLUX.DEV"
        elif "PONY" in base_model:
            return "PONY"
        elif "SDXL" in base_model or "XL" in base_model:
            return "SDXL"
        elif "SD 1.5" in base_model or "SD15" in base_model:
            return "SD15"
        elif "ILLUST" in base_model:
            return "ILLUST"

        # Check version name
        if "FLUX" in version_name:
            return "FLUX.DEV"
        elif "PONY" in version_name:
            return "PONY"
        elif "SDXL" in version_name or "XL" in version_name:
            return "SDXL"
        elif "SD 1.5" in version_name or "SD15" in version_name:
            return "SD15"
        elif "ILLUST" in version_name:
            return "ILLUST"

        return None

    def format_title(
        self,
        model_data: Dict[str, Any],
        specific_version_id: Optional[int] = None
    ) -> str:
        """Format title with base model prefix and cleaned name"""
        base_model = self.detect_base_model(
            model_data,
            specific_version_id
        )
        model_name = model_data.get("name", "Unknown Model")

        # Remove underscores from model name
        clean_name = model_name.replace("_", " ")

        # Apply title case (capitalize first letter of each word)
        clean_name = clean_name.title()

        # Remove invalid characters for filenames
        clean_name = re.sub(r'[\[\]/\\()]', '', clean_name)

        if base_model:
            title = f"{base_model} - {clean_name}"
        else:
            title = clean_name

        # Remove invalid characters from final title
        title = re.sub(r'[\[\]/\\()]', '', title)
        return title

    def generate_obsidian_tags(
        self,
        model_data: Dict[str, Any],
        specific_version_id: Optional[int] = None
    ) -> List[str]:
        """Generate Obsidian tags based on model type and base model"""
        base_model = self.detect_base_model(
            model_data,
            specific_version_id
        )
        model_type = model_data.get('type', 'Unknown').upper()
        model_name = model_data.get("name", "Unknown")

        # Clean model name for tag
        clean_tag_name = (
            model_name.lower()
            .replace("_", "-")
            .replace(" ", "-")
        )

        # Base tags from config
        metadata_config = self.config.get("metadata", {})
        tags = metadata_config.get(
            "base_tags",
            ["saga-of-gone", "ai", "civitai"]
        ).copy()

        # Check if it's a detailer based on tags or name
        is_detailer = False
        civitai_tags = model_data.get("tags", [])
        name_lower = model_name.lower()
        detailer_keywords = [
            "detail",
            "detailed",
            "enhancer",
            "detailer"
        ]
        if any(tag in detailer_keywords for tag in civitai_tags):
            is_detailer = True
        if any(keyword in name_lower for keyword in detailer_keywords):
            is_detailer = True

        # Add type-specific tags
        if model_type == 'LORA':
            tags.append("lora")

            # Add base model tag
            if base_model:
                tags.append(base_model.lower().replace(".", "-"))

            tags.append("diffusion")

            # Add detailer tag if applicable
            if is_detailer:
                tags.append("detailer")

        elif model_type == 'CHECKPOINT':
            # Add base model tag
            if base_model:
                tags.append(base_model.lower().replace(".", "-"))

            tags.append("diffusion")
            tags.append("diffusion-models")

        elif model_type == 'TEXTUALINVERSION':
            tags.append("embedding")
            if base_model:
                tags.append(base_model.lower().replace(".", "-"))
        else:
            # Generic fallback
            tags.append(model_type.lower())

        # Add model/lora name tag
        tags.append(clean_tag_name)

        # Return as list for YAML formatting
        return tags

    def format_generation_params(
        self,
        meta: Dict[str, Any]
    ) -> str:
        """Format generation parameters as markdown"""
        if not meta:
            return "_No generation parameters available_\n"

        # Key parameters to highlight
        important_keys = [
            "prompt", "negativePrompt", "Model", "sampler", "steps",
            "cfgScale", "seed", "Size", "Clip skip", "Hires upscale",
            "Hires upscaler", "Denoising strength"
        ]

        # Keys to exclude (massive metadata that bloats notes)
        excluded_keys = ["comfy"]

        sections = []

        # Prompt (if exists)
        if "prompt" in meta:
            prompt = meta["prompt"]
            sections.append(
                f"**Positive Prompt:**\n```\n{prompt}\n```\n"
            )

        # Negative prompt (if exists)
        if "negativePrompt" in meta:
            neg_prompt = meta["negativePrompt"]
            sections.append(
                f"**Negative Prompt:**\n```\n{neg_prompt}\n```\n"
            )

        # Other parameters
        other_params = []
        for key in important_keys:
            if key in meta and key not in ["prompt", "negativePrompt"]:
                other_params.append(f"- **{key}:** {meta[key]}")

        # Add any remaining params not in important_keys or excluded
        for key, value in meta.items():
            if key not in important_keys and key not in excluded_keys:
                other_params.append(f"- **{key}:** {value}")

        if other_params:
            sections.append(
                "**Parameters:**\n" + "\n".join(other_params)
            )

        return "\n\n".join(sections)

    # Extensions we consider when looking up the on-disk filename for
    # an image id, in priority order.
    _IMAGE_EXTENSIONS = ("jpeg", "jpg", "png", "webp", "gif")

    @classmethod
    def find_image_filename(
        cls,
        images_folder: Path,
        image_id: Any
    ) -> Optional[str]:
        """Return the actual on-disk filename for an image id, or None."""
        for ext in cls._IMAGE_EXTENSIONS:
            candidate = images_folder / f"{image_id}.{ext}"
            if candidate.exists():
                return candidate.name
        return None

    @classmethod
    def scan_image_ids_in_folder(cls, images_folder: Path) -> set[int]:
        """Return the set of numeric image IDs present in a folder.

        Image files are named `{id}.{ext}` — we treat the stem as an int
        and skip anything that doesn't parse, which filters out stray
        files a user may have dropped into the folder.
        """
        if not images_folder.is_dir():
            return set()
        ids: set[int] = set()
        for entry in images_folder.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lstrip('.').lower() not in cls._IMAGE_EXTENSIONS:
                continue
            try:
                ids.add(int(entry.stem))
            except ValueError:
                continue
        return ids

    # Embed pattern for image references inside the generated notes.
    # Allows optional `|alt-text` or `#anchor` suffixes that a user may
    # have added by hand. The `\.` is a literal dot — not `.\w+`, which
    # would also match a stray character before the extension.
    _EMBED_PATTERN = re.compile(
        r'!\[\[[^\]]*?/(\d+)\.\w+(?:[|#][^\]]*)?\]\]'
    )

    # Matches a complete YAML frontmatter block at the start of a file:
    # `---<EOL><body><EOL>---<EOL>`. Handles both LF and CRLF endings.
    # Group 1 captures the body between the fences.
    _FRONTMATTER_PATTERN = re.compile(
        r'\A---\r?\n(.*?)\r?\n---\r?\n', re.DOTALL
    )

    @classmethod
    def extract_image_ids_from_markdown(cls, content: str) -> set[int]:
        """Pull image IDs referenced by `![[...]]` embeds in a note.

        We deliberately union this with the on-disk scan: a user may
        have deleted an image file but kept its entry in the doc (or
        vice versa), and either signal means "we've seen this one".
        """
        return {int(m) for m in cls._EMBED_PATTERN.findall(content)}

    @classmethod
    def extract_frontmatter_field(
        cls,
        content: str,
        field: str
    ) -> Optional[str]:
        """Return the raw value of `field:` from frontmatter, or None.

        Used to sanity-check that an existing note actually corresponds
        to the model we're about to update — see the `source:` guard in
        the update flow. The value is returned stripped, with leading
        and trailing whitespace removed.
        """
        match = cls._FRONTMATTER_PATTERN.match(content)
        if not match:
            return None
        block = match.group(1)
        field_re = re.compile(
            rf'^{re.escape(field)}:\s*(.*)$', re.MULTILINE
        )
        field_match = field_re.search(block)
        if not field_match:
            return None
        return field_match.group(1).strip()

    @classmethod
    def upsert_frontmatter_field(
        cls,
        content: str,
        field: str,
        value: str
    ) -> str:
        """Set `field: value` in the YAML frontmatter, adding if needed.

        Assumes the frontmatter is the standard `---`-delimited block at
        the top of the file. If no frontmatter exists the content is
        returned unchanged — we don't want to invent one mid-update.
        The output always uses LF line endings inside the frontmatter
        and preserves everything after the closing fence byte-for-byte.
        """
        match = cls._FRONTMATTER_PATTERN.match(content)
        if not match:
            return content

        block = match.group(1)
        after_block = content[match.end():]
        replacement = f'{field}: {value}'

        field_pattern = re.compile(
            rf'^{re.escape(field)}:.*$', re.MULTILINE
        )
        if field_pattern.search(block):
            new_block = field_pattern.sub(
                lambda _m: replacement, block, count=1
            )
        else:
            # Slot the new field right after `created:` so related date
            # fields stay grouped. Fall back to appending if no
            # `created:` line exists.
            created_pattern = re.compile(r'^(created:.*)$', re.MULTILINE)
            if created_pattern.search(block):
                new_block = created_pattern.sub(
                    lambda m: f'{m.group(1)}\n{replacement}',
                    block,
                    count=1
                )
            else:
                new_block = block.rstrip('\r\n') + f'\n{replacement}'

        # Normalize: strip trailing newlines from the block so the
        # closing fence doesn't end up with a blank line before it.
        new_block = new_block.rstrip('\r\n')
        return f'---\n{new_block}\n---\n{after_block}'

    @classmethod
    def find_notes_by_source(
        cls,
        note_dir: Path,
        expected_source: str,
    ) -> List[Path]:
        """Find notes in `note_dir` whose frontmatter `source:` matches.

        Fallback for update mode when the user has renamed a note away
        from the computed default filename (e.g. to add a version or
        print suffix). The `source:` field is the authoritative pointer
        back to CivitAI and survives renames.

        Non-recursive: only scans `note_dir` directly. Returns every
        match so callers can detect ambiguity (e.g. two notes claiming
        the same model) rather than silently picking one.
        """
        if not note_dir.is_dir():
            return []
        target = expected_source.rstrip('/')
        matches: List[Path] = []
        for md_file in sorted(note_dir.glob('*.md')):
            try:
                content = md_file.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                continue
            source = cls.extract_frontmatter_field(content, 'source')
            if source is None:
                continue
            if source.rstrip('/') == target:
                matches.append(md_file)
        return matches

    def build_update_section(
        self,
        new_images: List[Dict[str, Any]],
        images_folder: Path,
        update_date: str
    ) -> str:
        """Render the markdown for an update batch of new images.

        Mirrors the layout of `generate_page`'s example-images section
        so updates look visually identical to the originals, just under
        a dated heading.
        """
        lines: List[str] = []
        lines.append(f"## Example Images — Update {update_date}\n")

        media_rel = self.config.get("obsidian", {}).get(
            "media_folder",
            "zzMedia/Model and Lora Example Images"
        )

        for idx, image_data in enumerate(new_images, 1):
            image_id = image_data.get("id", idx)
            image_filename = self.find_image_filename(
                images_folder, image_id
            ) or f"{image_id}.jpeg"

            lines.append(f"#### Image {idx}\n")
            relative_path = (
                f"{media_rel}/{images_folder.name}/{image_filename}"
            )
            lines.append(f"![[{relative_path}]]\n")

            stats = image_data.get("stats", {})
            reactions = (
                stats.get("likeCount", 0) + stats.get("heartCount", 0)
            )
            width = image_data.get("width")
            height = image_data.get("height")

            stats_parts = []
            if reactions > 0:
                stats_parts.append(f"{reactions} reactions")
            if width and height:
                stats_parts.append(f"{width}×{height}")
            if stats_parts:
                lines.append(f"*{' | '.join(stats_parts)}*\n")

            meta = image_data.get("meta")
            if meta and isinstance(meta, dict):
                lines.append(self.format_generation_params(meta))
            else:
                lines.append("_No generation parameters available_")

            lines.append("\n---\n")

        return "\n".join(lines)

    def generate_page(
        self,
        model_data: Dict[str, Any],
        images_data: List[Dict[str, Any]],
        images_folder: Path,
        model_name: str,
        specific_version_id: Optional[int] = None
    ) -> str:
        """Generate the complete Obsidian markdown page"""
        from datetime import datetime

        lines = []

        # Get formatted title
        page_title = self.format_title(model_data, specific_version_id)

        # Generate Obsidian tags
        obsidian_tags = self.generate_obsidian_tags(
            model_data,
            specific_version_id
        )

        # Get author from config
        metadata_config = self.config.get("metadata", {})
        author = metadata_config.get("author", "Unknown")

        # Metadata section (YAML frontmatter)
        lines.append("---")
        lines.append("tags:")
        for tag in obsidian_tags:
            lines.append(f"  - {tag}")
        lines.append(f"author: {author}")
        lines.append(f"created: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(
            f"source: https://civitai.com/models/{model_data.get('id')}"
        )
        lines.append(f"type: {model_data.get('type', 'Unknown')}")

        creator = model_data.get('creator', {}).get(
            'username',
            'Unknown'
        )
        lines.append(f"civitai creator: {creator}")

        stats = model_data.get('stats', {})
        downloads = stats.get('downloadCount', 'N/A')
        if downloads != 'N/A':
            lines.append(f"downloads: {downloads:,}")
        else:
            lines.append(f"downloads: {downloads}")

        rating = stats.get('rating', 'N/A')
        rating_line = (
            f"rating: {rating}/5" if rating != 'N/A' else "Rating: N/A"
        )
        lines.append(rating_line)

        # Upload date from first version
        versions = model_data.get("modelVersions", [])
        if versions:
            upload_date = versions[0].get("createdAt", "N/A")
            if upload_date != "N/A":
                # Format ISO date to readable format
                try:
                    date_obj = datetime.fromisoformat(
                        upload_date.replace('Z', '+00:00')
                    )
                    upload_date = date_obj.strftime('%Y-%m-%d')
                except (ValueError, TypeError, AttributeError):
                    # If parsing fails or upload_date is not a string
                    pass
            lines.append(f"upload date: {upload_date}")

        # CivitAI tags
        civitai_tags = model_data.get("tags", [])
        if civitai_tags:
            lines.append(f"civitai tags: {', '.join(civitai_tags)}")
        lines.append("---")
        lines.append("")

        # Title
        lines.append(f"# {page_title}\n")

        # Description
        description = model_data.get("description", "")
        if description:
            lines.append("## Description\n")
            # Strip HTML tags from description
            clean_desc = re.sub(r'<[^>]+>', '', description)
            lines.append(f"{clean_desc}\n")

        lines.append("---\n")

        # Images section
        lines.append("## Example Images\n")

        for idx, image_data in enumerate(images_data, 1):
            image_id = image_data.get("id", idx)

            # Look up the actual file on disk so we use the right
            # extension (the CDN serves PNG/WEBP/JPEG interchangeably).
            # Fall back to .jpeg only when nothing was downloaded —
            # e.g. running with --skip-download for a dry preview.
            image_filename = self.find_image_filename(
                images_folder, image_id
            ) or f"{image_id}.jpeg"

            lines.append(f"#### Image {idx}\n")

            # Image embed - using relative path from vault root
            media_rel = self.config.get("obsidian", {}).get(
                "media_folder",
                "zzMedia/Model and Lora Example Images"
            )
            relative_path = (
                f"{media_rel}/{images_folder.name}/{image_filename}"
            )
            lines.append(f"![[{relative_path}]]\n")

            # Image stats
            stats = image_data.get("stats", {})
            reactions = (
                stats.get("likeCount", 0) + stats.get("heartCount", 0)
            )
            width = image_data.get("width")
            height = image_data.get("height")

            # Build stats line
            stats_parts = []
            if reactions > 0:
                stats_parts.append(f"{reactions} reactions")
            if width and height:
                stats_parts.append(f"{width}×{height}")

            if stats_parts:
                lines.append(f"*{' | '.join(stats_parts)}*\n")

            # Generation parameters
            meta = image_data.get("meta")
            if meta and isinstance(meta, dict):
                lines.append(self.format_generation_params(meta))
            else:
                lines.append("_No generation parameters available_")

            lines.append("\n---\n")

        return "\n".join(lines)

    def save_page(
        self,
        content: str,
        filename: str,
        model_data: Dict[str, Any],
        specific_version_id: Optional[int] = None
    ) -> str:
        """Save the markdown page to the vault and return the filename"""
        # Get the appropriate directory for this note
        note_dir = self.get_note_directory(
            model_data,
            specific_version_id
        )
        note_dir.mkdir(parents=True, exist_ok=True)

        output_path = note_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n✅ Obsidian page saved to: {output_path}")
        return filename


# --------------------------------------------------------------------- #
# Interactive mode
# --------------------------------------------------------------------- #
#
# Triggered by --interactive / -i. Two pieces:
#   1. run_config_wizard() — first-run setup when config.yaml is missing
#   2. run_interactive_flow() — replaces argparse-driven options with
#      arrow-key prompts, and offers a real version picker fetched from
#      the API (the main thing you can't get from a static bash wrapper)
#
# Everything here mutates the same argparse.Namespace that the CLI path
# produces, so the downstream logic in main() doesn't need to branch on
# how the options were collected.


def _require_questionary() -> Any:
    """Return the questionary module or exit with an install hint.

    Kept as a function (rather than a module-level guard) so the script
    still runs for users who never pass --interactive and haven't pip
    installed the new dep.
    """
    if questionary is None:
        print(
            "❌ Interactive mode needs `questionary`. Install with:\n"
            "     pip install questionary\n"
            "   or re-run `pip install -r requirements.txt`."
        )
        sys.exit(1)
    return questionary


def _ask_or_exit(prompt: Any) -> Any:
    """Run a questionary prompt; exit cleanly if the user cancels.

    `.ask()` returns None on Ctrl+C / Esc for every prompt type, which
    is ambiguous with a legitimate False from a confirm — so we always
    treat None as "user bailed out" and exit 0 rather than crashing
    deeper in the flow with an AttributeError.
    """
    result = prompt.ask()
    if result is None:
        print("\n👋 Cancelled.")
        sys.exit(0)
    return result


def run_config_wizard(config_path: str) -> None:
    """Walk the user through creating a config.yaml from scratch.

    Uses config.example.yaml as the baseline so we inherit every comment
    and sensible default, and only overlay the fields the user provided.
    The output is written via yaml.safe_dump, which drops the comments —
    that's a known tradeoff for keeping the wizard simple; the example
    file stays in the repo as a reference.
    """
    q = _require_questionary()

    print("\n👋 No config file found — let's set one up.\n")

    example_path = Path(__file__).parent / "config.example.yaml"
    if example_path.exists():
        with open(example_path, 'r', encoding='utf-8') as f:
            cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # Vault path: we only hard-reject paths that exist but aren't
    # directories (a file or symlink at the vault location is almost
    # certainly a typo). Missing paths are accepted with a confirm —
    # legitimate when the user is creating a new vault, or pointing at
    # a directory they're about to mount/sync.
    def _validate_dir(p: str) -> Any:
        if not p.strip():
            return "Required"
        expanded = Path(p).expanduser()
        if expanded.exists() and not expanded.is_dir():
            return f"Path exists but is not a directory: {expanded}"
        return True

    while True:
        vault_raw = _ask_or_exit(q.path(
            "Path to your Obsidian vault:",
            only_directories=True,
            validate=_validate_dir
        ))
        expanded_vault = Path(vault_raw).expanduser()
        if expanded_vault.exists():
            break
        # Missing path — likely correct (new vault) but plausibly a
        # typo, so make the user confirm rather than silently accepting.
        if _ask_or_exit(q.confirm(
            f"'{expanded_vault}' doesn't exist yet. Use it anyway?",
            default=False
        )):
            break
        # User said no — loop back and let them re-enter.
    cfg.setdefault("obsidian", {})["vault_path"] = str(expanded_vault)

    author = _ask_or_exit(q.text(
        "Your name (used in the `author:` frontmatter field):",
        default=cfg.get("metadata", {}).get("author", "Your Name")
    ))
    cfg.setdefault("metadata", {})["author"] = author or "Your Name"

    api_key = _ask_or_exit(q.password(
        "CivitAI API key (optional — hit Enter to skip):"
    ))
    if api_key:
        cfg.setdefault("civitai", {})["api_key"] = api_key

    base_dir_default = cfg.get("obsidian", {}).get(
        "base_directory", "Diffusion"
    )
    base_dir = _ask_or_exit(q.text(
        "Base directory in vault for notes:",
        default=base_dir_default
    ))
    cfg.setdefault("obsidian", {})["base_directory"] = (
        base_dir or base_dir_default
    )

    output = Path(config_path)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_yaml_write(output, cfg)
    except Exception as exc:
        # Disk full, permission denied, weird FS state — the user has
        # just answered six prompts. Don't make them re-do that work:
        # dump the YAML to stdout so they can paste it themselves.
        print(
            f"\n❌ Couldn't write to {output}: {exc}\n"
            "   Your answers are below — paste this into the file "
            "manually and re-run:\n"
        )
        print("---")
        print(yaml.safe_dump(
            cfg, sort_keys=False, default_flow_style=False
        ).rstrip())
        print("---")
        sys.exit(1)

    print(f"\n✅ Config written to: {output}")
    print(
        "   You can edit it directly any time — see config.example.yaml "
        "for the full set of options.\n"
    )


def _atomic_yaml_write(path: Path, data: Dict[str, Any]) -> None:
    """Write a YAML dict atomically: stage in tmp, fsync, rename.

    Mirrors the update-mode write in main() — a crash mid-write should
    never leave config.yaml truncated, since that would brick the next
    run with a YAML parse error. `Path.replace` is atomic on POSIX and
    overwrites on Windows.
    """
    tmp = path.with_name(f'.{path.name}.tmp')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                data, f, sort_keys=False, default_flow_style=False
            )
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def save_to_config(
    config_path: str,
    updates: Dict[str, Dict[str, Any]]
) -> None:
    """Merge sectioned updates into an existing config.yaml.

    `updates` maps top-level section name → field dict, e.g.
    ``{'defaults': {'sort_order': 'Newest'}, 'civitai': {'api_key': '…'}}``.
    Sections that don't exist yet are created.

    Comments in config.yaml are lost on round-trip (yaml.safe_dump
    doesn't preserve them). Callers should warn the user up front so
    they can opt out — see how run_interactive_flow uses this.
    """
    path = Path(config_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        cfg = {}
    for section, fields in updates.items():
        cfg.setdefault(section, {}).update(fields)
    _atomic_yaml_write(path, cfg)


def _apply_cli_overrides_to_config(
    config: Dict[str, Any],
    args: argparse.Namespace
) -> None:
    """Fold flag-level overrides into the loaded config dict.

    Run once, before any code consumes the config — both the
    interactive flow's status header and main()'s execution path
    should see the same effective values, regardless of whether the
    user provided them on the CLI or in YAML.

    Only the overrides that affect either the interactive flow's
    display/decisions or downstream config consumers live here; per-run
    knobs that main() reads off ``args`` directly (delay, api-delay,
    sort, period, limit, nsfw, etc.) are intentionally not mirrored
    into config — the args namespace is their canonical home.
    """
    if args.vault_path:
        config.setdefault("obsidian", {})["vault_path"] = args.vault_path
    if args.api_key:
        config.setdefault("civitai", {})["api_key"] = args.api_key


def _build_fetcher(
    config: Dict[str, Any],
    api_key_override: Optional[str] = None
) -> CivitAIFetcher:
    """Construct a CivitAIFetcher from config, with an optional override.

    Single source of truth for fetcher wiring so the interactive flow
    and main() stay in sync — previously they each instantiated the
    fetcher independently, and the interactive version didn't honor
    ``args.api_key``. The override wins when set so a CLI flag still
    beats whatever the YAML carries.
    """
    civitai_cfg = config.get("civitai", {})
    rate_limits = config.get("rate_limits", {})
    return CivitAIFetcher(
        api_key=api_key_override or civitai_cfg.get("api_key"),
        base_url=civitai_cfg.get(
            "base_url", "https://civitai.com/api/v1"
        ),
        max_retries=rate_limits.get("max_retries", 3),
        backoff_factor=rate_limits.get("backoff_factor", 1)
    )


class TargetPaths(NamedTuple):
    """The note file and image folder a run will read from or write to."""
    note_path: Path
    images_folder: Path


def compute_target_paths(
    generator: ObsidianPageGenerator,
    model_data: Dict[str, Any],
    version_id: Optional[int]
) -> TargetPaths:
    """Compute the (note, images folder) a run targets.

    Single source of truth for path resolution — previously inlined in
    both main() and run_interactive_flow, where the two copies had
    already started drifting on error-message indentation. A fresh
    fetch and an update both land at exactly these paths.
    """
    formatted_title = generator.format_title(model_data, version_id)
    note_dir = generator.get_note_directory(model_data, version_id)
    note_path = note_dir / f"{formatted_title}.md"
    folder_name = ObsidianPageGenerator.build_image_folder_name(
        model_data, version_id
    )
    return TargetPaths(note_path, generator.media_folder / folder_name)


class UpdatePreflight(NamedTuple):
    """Result of running the update-mode pre-flight checks.

    `problems` is non-empty when the run must abort. `warning` is
    informational only — printed but doesn't stop the run.
    `existing_content` is the note's text when read successfully (used
    by main() to compute already-known image IDs); None when the
    check bailed before reading the file.
    `resolved_note_path` is the path of the note we actually located
    — equal to the computed default in the common case, but may
    differ when the user renamed the note and we fell back to a
    `source:` frontmatter match. Callers must write back to this
    path, not the original computed path.
    """
    problems: List[str]
    warning: Optional[str]
    existing_content: Optional[str]
    resolved_note_path: Optional[Path] = None


def check_update_preflight(
    paths: TargetPaths,
    expected_source: str
) -> UpdatePreflight:
    """Verify an update can run against the given target paths.

    Two checks run in priority order:

    1. Both the note and the image folder must exist and be the right
       kind of filesystem object. If not, problems are collected and
       returned without reading the file — there's nothing to verify
       yet.
    2. If the file is readable, its `source:` frontmatter must match
       the model we're about to update. Two different models can
       produce notes with the same title (e.g. both LoRAs named
       "Style Test" on SDXL); the source field is the authoritative
       pointer back to CivitAI.

    Returning a result struct (rather than printing + sys.exit) lets
    each caller render its own error UX — the interactive flow uses
    a different leading message than main() does, since the user's
    next action ("re-run without --update" vs "answer the prompt
    differently") differs by context.
    """
    problems: List[str] = []

    # Resolve the note path. Fast path = computed default. If the
    # user has renamed the note (e.g. added a version or print
    # suffix), fall back to scanning the note's directory for a
    # markdown file whose frontmatter `source:` points at this
    # model. The source field carries the model id and survives
    # renames, making it the authoritative pointer.
    #
    # Exactly one of the three branches below adds a problem (or
    # sets the resolved path), so callers never see compound or
    # contradictory note-resolution errors.
    resolved_note_path: Optional[Path] = None
    if paths.note_path.exists():
        if paths.note_path.is_file():
            resolved_note_path = paths.note_path
        else:
            problems.append(
                f"Path exists but is not a regular file:\n"
                f"      {paths.note_path}"
            )
    else:
        candidates = ObsidianPageGenerator.find_notes_by_source(
            paths.note_path.parent, expected_source
        )
        if len(candidates) == 1:
            resolved_note_path = candidates[0]
        elif len(candidates) > 1:
            listed = '\n        '.join(str(p) for p in candidates)
            problems.append(
                "Multiple notes in this folder claim the same "
                f"`source:` URL ({expected_source}).\n"
                "      Refusing to guess which one to update — "
                "please remove or rename the duplicates so exactly "
                "one remains:\n"
                f"        {listed}"
            )
        else:
            problems.append(
                f"Obsidian note not found at:\n      {paths.note_path}\n"
                "      Also searched the surrounding folder for a "
                f"note with `source: {expected_source}` in its "
                "frontmatter and found none."
            )

    if not paths.images_folder.exists():
        problems.append(
            f"Image folder not found at:\n      "
            f"{paths.images_folder}"
        )
    elif not paths.images_folder.is_dir():
        problems.append(
            f"Image folder path exists but is not a directory:\n"
            f"      {paths.images_folder}"
        )

    if problems:
        return UpdatePreflight(problems, None, None)

    # No problems means the resolution branch above produced a path.
    assert resolved_note_path is not None
    existing_content = resolved_note_path.read_text(encoding='utf-8')
    existing_source = (
        ObsidianPageGenerator.extract_frontmatter_field(
            existing_content, 'source'
        )
    )

    warning: Optional[str] = None
    if existing_source is None:
        warning = (
            "Existing note has no `source:` field in its frontmatter "
            "— skipping model-match verification. If this note wasn't "
            "generated by this script, double-check the path is right "
            "before proceeding."
        )
    elif existing_source.rstrip('/') != expected_source:
        # Can't trigger when we located the note via source-match
        # fallback (we found it BY this exact source), but the
        # check protects the fast-path branch where the file at
        # the computed path may belong to a different model whose
        # title happens to collide.
        problems.append(
            "The existing note at this path belongs to a different "
            "model.\n"
            f"      Expected source: {expected_source}\n"
            f"      Note's source:   {existing_source}\n"
            "      Refusing to update — appending here would corrupt "
            "that other note."
        )
        return UpdatePreflight(problems, None, None)

    return UpdatePreflight(
        [], warning, existing_content, resolved_note_path
    )


def check_fresh_fetch_safety(
    note_path: Path,
    expected_source: str,
) -> List[str]:
    """Return problem messages if a fresh fetch would clobber state.

    Two scenarios are detected, in priority order:

    1. One or more notes in `note_path.parent` already carry this
       model's `source:` URL. That catches both the "default path is
       still occupied" case and the "user renamed the note" case in
       a single query — the source field is the authoritative model
       id pointer and survives renames. A blind fresh fetch would
       either wipe the user's manual edits (default path) or
       silently create a duplicate alongside the renamed one. Either
       outcome is almost never what the user wants; better to refuse
       and steer them to `--update`.

    2. A note already sits at the exact computed default path but
       belongs to a DIFFERENT model (or has no `source:` field to
       verify against). This is the most dangerous scenario — a
       blind save_page would overwrite an unrelated note. Refuse
       unconditionally; the user can move/rename the occupant.

    An empty list means "safe to proceed" — no existing notes block
    the write. Returning a list (rather than printing + sys.exit)
    keeps this helper unit-testable and lets callers compose their
    own error UX.
    """
    problems: List[str] = []
    same_source_notes = ObsidianPageGenerator.find_notes_by_source(
        note_path.parent, expected_source
    )
    if same_source_notes:
        listed = '\n   • '.join(str(p) for p in same_source_notes)
        problems.append(
            "A note for this model already exists:\n"
            f"   • {listed}\n\n"
            "   To add more images to it, re-run with --update.\n"
            "   To regenerate from scratch, delete the existing "
            "note(s) first."
        )
        return problems

    if note_path.exists():
        try:
            other_source = (
                ObsidianPageGenerator.extract_frontmatter_field(
                    note_path.read_text(encoding='utf-8'), 'source'
                )
            )
        except OSError:
            other_source = None
        lines = [
            "A note already occupies the target path:",
            f"   {note_path}",
        ]
        if other_source:
            lines.append(f"   That note's source: {other_source}")
            lines.append(f"   This model's source: {expected_source}")
        else:
            lines.append(
                "   That note has no `source:` field, so we can't "
                "verify it belongs to a different model — refusing "
                "to overwrite blind."
            )
        lines.append(
            "\n   Move, rename, or delete that note first if you "
            "want this model's note to take its place."
        )
        problems.append('\n'.join(lines))

    return problems


def _build_version_choice(version: Dict[str, Any]) -> Any:
    """Render one model-version entry as a questionary choice.

    Pulls the human name, base model, and id into a single label like
    "v1.2 [SDXL] (id 12345)" so the picker is informative even when
    multiple versions share a similar name.
    """
    q = _require_questionary()
    name = version.get("name") or "Unnamed"
    base = version.get("baseModel") or "?"
    vid = version.get("id")
    return q.Choice(f"{name} [{base}] (id {vid})", value=vid)


def run_interactive_flow(
    config: Dict[str, Any],
    args: argparse.Namespace
) -> argparse.Namespace:
    """Populate `args` via prompts and return the same namespace.

    Mutates the namespace produced by argparse so the downstream logic
    in main() doesn't have to know whether values came from flags or
    prompts. Does one early API call to fetch model details — this is
    what lets us offer a real version picker, which is the main reason
    we picked questionary over a bash/Gum wrapper.
    """
    q = _require_questionary()
    defaults = config.get("defaults", {})

    # 0a) Status header — three lines of orientation so the user
    #     doesn't have to Ctrl+C and `cat config.yaml` to remember
    #     which vault they're targeting.
    obsidian_cfg = config.get("obsidian", {})
    civitai_cfg = config.get("civitai", {})
    api_key_status = (
        "set" if civitai_cfg.get("api_key") else "not set"
    )
    print(f"\n📂 Config:    {args.config}")
    print(
        f"🗄️  Vault:     {obsidian_cfg.get('vault_path', '(not set)')}"
    )
    print(f"🔑 API key:   {api_key_status}\n")

    # 0b) Offer to add an API key when none is configured. The
    #     unauthenticated rate limit (~100 req/min) is enough for a
    #     small run but routinely throttles a 200-image fetch, so
    #     pointing this out up front saves the user from discovering
    #     it mid-run when a 429 backoff stretches a job to 15 minutes.
    if not civitai_cfg.get("api_key"):
        print(
            "ℹ️  No CivitAI API key configured — you'll get the lower "
            "unauthenticated\n   rate limits. Get one at: "
            "https://civitai.com/user/account"
        )
        if _ask_or_exit(q.confirm(
            "Enter an API key for this run?",
            default=False
        )):
            new_key = _ask_or_exit(q.password(
                "CivitAI API key:"
            )).strip()
            if new_key:
                # Mutate the in-memory config so the fetcher built
                # below (which re-reads from config) picks it up;
                # main() also reads from the same config dict, so a
                # single assignment covers both the prefetch and the
                # actual run.
                config.setdefault("civitai", {})["api_key"] = new_key
                if _ask_or_exit(q.confirm(
                    "Save the key to config.yaml for future runs?",
                    default=True
                )):
                    # save_to_config strips comments via safe_dump —
                    # the API key is a secret-ish value the user
                    # almost certainly wants persisted, so warning
                    # would be more annoying than helpful here. The
                    # heavier warning lives on the save-defaults
                    # prompt below where the choice is more optional.
                    try:
                        save_to_config(
                            args.config,
                            {"civitai": {"api_key": new_key}}
                        )
                        print(f"✅ Saved to {args.config}\n")
                    except Exception as exc:
                        print(
                            f"⚠️  Couldn't save to {args.config}: "
                            f"{exc}\n   The key will be used for "
                            f"this run only.\n"
                        )

    # 1) Model URL / ID
    def _validate_model(s: str) -> Any:
        if not s.strip():
            return "Required"
        # Use the same extractor the rest of the script uses so the
        # validation surface matches: anything CivitAIFetcher accepts is
        # fine here, anything it would reject we reject up front. Called
        # as a staticmethod so we don't build a session/retry adapter
        # purely to validate a string.
        try:
            CivitAIFetcher.extract_model_id(s.strip())
        except ValueError as exc:
            return str(exc)
        return True

    model_input = _ask_or_exit(q.text(
        "CivitAI model URL or ID:",
        validate=_validate_model
    )).strip()

    # 2) Prefetch model_data so we can show name/type and populate the
    #    version picker with real choices. This is one wasted API call
    #    relative to the run that follows, but it's worth it for the
    #    UX win — the user gets to confirm "yes, that's the right
    #    model" before committing to any downloads.
    # CLI overrides have already been folded into `config` by
    # _apply_cli_overrides_to_config, so the fetcher picks up
    # --api-key without any extra wiring here.
    fetcher = _build_fetcher(config)

    model_id, version_from_url = fetcher.extract_model_id(model_input)
    print("\nFetching model details...")
    try:
        model_data = fetcher.get_model_details(model_id)
    except Exception as exc:
        print(f"❌ Couldn't fetch model {model_id}: {exc}")
        sys.exit(1)

    name = model_data.get("name", f"model_{model_id}")
    mtype = model_data.get("type", "Unknown")
    versions = model_data.get("modelVersions", []) or []
    print(
        f"✓ Found: {name} ({mtype}) — {len(versions)} version(s) available"
    )

    # 3) Version picker — only meaningful when the URL didn't already
    #    pin one and the model has more than one version on file.
    chosen_version_id: Optional[int] = version_from_url
    if version_from_url is None and len(versions) > 1:
        choices = [
            q.Choice("All versions (let the script iterate)", value=None)
        ] + [_build_version_choice(v) for v in versions]
        chosen_version_id = _ask_or_exit(q.select(
            "Which version do you want images for?",
            choices=choices
        ))

    # Rebuild the model arg with the chosen version embedded so the
    # downstream extract_model_id() call sees the same thing whether
    # we got here interactively or via the CLI.
    if chosen_version_id is not None:
        args.model = (
            f"https://civitai.com/models/{model_id}"
            f"?modelVersionId={chosen_version_id}"
        )
    else:
        args.model = str(model_id)

    # 4) Update mode — asked early because:
    #    (a) its answer drives the defaults for the sort/period prompts
    #        that follow (CLI mode does the same swap at lines further
    #        down in main()), and
    #    (b) when update is on we run the same pre-flight checks main()
    #        runs, so the user discovers a missing note immediately
    #        rather than after walking through every other prompt.
    #
    # Skipped when --update is already on the CLI — the user has been
    # explicit and we shouldn't pretend they haven't been.
    if not args.update:
        args.update = _ask_or_exit(q.confirm(
            "Update an existing note (append new images) instead of "
            "generating fresh?",
            default=False
        ))

    # Mutual-exclusion guard runs *before* the rest of the prompts so
    # a user who passed --skip-download on the CLI and then picks
    # update interactively finds out now, not after answering 5 more
    # questions and the summary screen.
    if args.update and args.skip_download:
        print(
            "\n❌ --update and --skip-download cannot be combined.\n"
            "   You passed --skip-download but chose update mode here."
        )
        sys.exit(1)

    # 5) Pre-flight when updating: run the same checks main() runs, in
    #    the same order. Bailing out here saves the user from walking
    #    through the rest of the prompts only to hit "note not found"
    #    or "wrong model" deeper in the run. The helper is the single
    #    source of truth — main() calls it too.
    preflight_generator = ObsidianPageGenerator(config=config)
    preflight_paths = compute_target_paths(
        preflight_generator, model_data, chosen_version_id
    )
    expected_source = f"https://civitai.com/models/{model_id}"

    if args.update:
        preflight = check_update_preflight(
            preflight_paths, expected_source
        )
        if preflight.problems:
            print(
                "\n❌ Update mode can't run — fix the following first:"
            )
            for p in preflight.problems:
                print(f"   • {p}")
            print(
                "\n   Re-run without update mode to do an initial "
                "fetch, or verify the model name / version ID matches "
                "what was used originally."
            )
            sys.exit(1)
        if preflight.warning:
            print(f"\n⚠️  {preflight.warning}")
        # If the preflight resolved to a renamed file, surface that
        # now so the user can bail before answering more prompts if
        # the match doesn't look right to them.
        if (
            preflight.resolved_note_path is not None
            and preflight.resolved_note_path != preflight_paths.note_path
        ):
            print(
                "\nℹ️  Note located via `source:` frontmatter "
                "(renamed from default):\n"
                f"   {preflight.resolved_note_path}"
            )
    else:
        # Fresh-fetch safety: mirror the same early-bail discipline
        # so an interactive user discovers a same-source / wrong-
        # model conflict here, before answering the remaining
        # prompts. main() runs the identical check again — that's
        # the source of truth — but doing it here saves them
        # from walking through five more questions for nothing.
        safety_problems = check_fresh_fetch_safety(
            preflight_paths.note_path, expected_source
        )
        if safety_problems:
            print("\n❌ Fresh fetch can't run — fix the following first:")
            for p in safety_problems:
                print(f"\n   {p}")
            sys.exit(1)

    # 6) Sort / period — defaults swap based on update mode to match
    #    the CLI behavior: an update is looking for what's *new*, so
    #    Newest/Month is a more useful starting point than the
    #    Most Reactions/AllTime defaults used for fresh fetches.
    if args.update:
        sort_default = "Newest"
        period_default = "Month"
    else:
        sort_default = defaults.get("sort_order", "Most Reactions")
        period_default = defaults.get("time_period", "AllTime")

    # Each of the remaining prompts is skipped when the user already
    # provided the value on the CLI — `-i --sort Newest --limit 50`
    # should ask only for what's left, not silently overwrite the
    # explicit choices. argparse defaults the value flags to None,
    # which is our "ask the user" signal.
    if args.sort is None:
        args.sort = _ask_or_exit(q.select(
            "Sort order:",
            choices=[
                "Most Reactions", "Most Comments", "Most Collected",
                "Newest", "Oldest",
            ],
            default=sort_default
        ))

    if args.period is None:
        args.period = _ask_or_exit(q.select(
            "Time period:",
            choices=["AllTime", "Year", "Month", "Week", "Day"],
            default=period_default
        ))

    # 7) NSFW
    if args.nsfw is None:
        nsfw_default = defaults.get("nsfw_filter", "all")
        args.nsfw = _ask_or_exit(q.select(
            "NSFW filter:",
            choices=[
                q.Choice("All (no filter)", value="all"),
                q.Choice("SFW only (block NSFW)", value="block"),
                q.Choice("NSFW only", value="allow"),
            ],
            default=nsfw_default
        ))

    # 8) Limit — text input with a real integer validator
    if args.limit is None:
        def _validate_limit(s: str) -> Any:
            if not s.isdigit():
                return "Enter a positive integer"
            if int(s) <= 0:
                return "Must be greater than zero"
            return True

        limit_str = _ask_or_exit(q.text(
            "Maximum number of images to fetch:",
            default=str(defaults.get("image_limit", 200)),
            validate=_validate_limit
        ))
        args.limit = int(limit_str)

    # 9) Quality filters — confirm prompts with True defaults to match
    #    the script's "ship a curated library by default" stance.
    #    BooleanOptionalAction makes --require-meta / --no-require-meta
    #    set the value to True/False; only None means "not specified".
    if args.require_meta is None:
        args.require_meta = _ask_or_exit(q.confirm(
            "Require generation metadata? (drops meta-less duds)",
            default=_as_bool(defaults.get("require_meta"), True)
        ))
    if args.filter_begging is None:
        args.filter_begging = _ask_or_exit(q.confirm(
            "Filter out 'buzz please' / tip-begging prompts?",
            default=_as_bool(defaults.get("filter_begging"), True)
        ))

    # 10) Skip-download — mutually exclusive with update mode, so the
    #     prompt only fires when update is off AND the CLI didn't
    #     already set --skip-download. When update is on we force the
    #     flag to False (the mutual-exclusion conflict for
    #     CLI-provided --skip-download was already caught above).
    if args.update:
        args.skip_download = False
    elif not args.skip_download:
        args.skip_download = _ask_or_exit(q.confirm(
            "Skip downloads (write the .md only)?",
            default=False
        ))

    # 11) Stash the prefetched model_data on the namespace so main()
    #     can reuse it instead of fetching the same model twice. Using
    #     an underscore-prefixed attribute to signal it's a private
    #     side-channel between the interactive flow and main().
    args._prefetched_model_data = model_data

    # 12) Final summary so the user can sanity-check before any
    #     rate-limited API work kicks off.
    print("\n📋 Summary")
    print(f"   Model:        {name} (ID {model_id})")
    if chosen_version_id is not None:
        version = next(
            (v for v in versions if v.get("id") == chosen_version_id),
            None
        )
        version_label = (
            version.get("name") if version else str(chosen_version_id)
        )
        print(f"   Version:      {version_label}")
    else:
        print("   Version:      (all)")
    print(f"   Update mode:  {args.update}")
    print(f"   Sort:         {args.sort}")
    print(f"   Period:       {args.period}")
    print(f"   NSFW:         {args.nsfw}")
    print(f"   Limit:        {args.limit}")
    print(f"   Require meta: {args.require_meta}")
    print(f"   Filter beg:   {args.filter_begging}")
    print(f"   Skip dl:      {args.skip_download}")
    print()

    # 13) Offer to persist the run options as the new config defaults.
    #     Update mode + skip-download are deliberately *not* saved —
    #     they're per-run intentions, not standing preferences, and
    #     persisting them would surprise the user on the next fresh
    #     fetch (e.g. saving `update: true` would silently turn every
    #     subsequent run into an update). Model and version are also
    #     excluded for the obvious reason.
    if _ask_or_exit(q.confirm(
        "Save these options (sort, period, NSFW, limit, filters) as "
        "the new defaults in config.yaml?",
        default=False
    )):
        # yaml.safe_dump loses comments — flag this so a user who has
        # invested in commenting their config can back out. The example
        # file stays in the repo as a reference, so the docs aren't
        # really lost, but the user's personal comments will be.
        print(
            "\n⚠️  This will rewrite config.yaml and any comments in "
            "it will be stripped.\n   (config.example.yaml stays "
            "intact as a reference.)"
        )
        if _ask_or_exit(q.confirm(
            "Proceed with the save?",
            default=True
        )):
            try:
                save_to_config(args.config, {
                    "defaults": {
                        "image_limit": args.limit,
                        "sort_order": args.sort,
                        "time_period": args.period,
                        "nsfw_filter": args.nsfw,
                        "require_meta": args.require_meta,
                        "filter_begging": args.filter_begging,
                    }
                })
                print(f"✅ Defaults saved to {args.config}\n")
            except Exception as exc:
                print(
                    f"⚠️  Couldn't save to {args.config}: {exc}\n"
                    "   Continuing with this run — your settings "
                    "weren't persisted.\n"
                )

    proceed = _ask_or_exit(q.confirm("Proceed?", default=True))
    if not proceed:
        print("👋 Cancelled.")
        sys.exit(0)

    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch CivitAI model images and create Obsidian docs"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help=(
            "CivitAI model URL or ID. Optional when --interactive is set "
            "(you'll be prompted for it)."
        )
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help=(
            "Drive the tool through arrow-key prompts instead of flags. "
            "Includes a real version picker fetched from the API and a "
            "first-run config wizard if config.yaml is missing."
        )
    )
    parser.add_argument(
        "--api-key",
        help="CivitAI API key for higher rate limits",
        default=None
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to fetch"
    )
    parser.add_argument(
        "--sort",
        choices=[
            "Most Reactions", "Most Comments", "Most Collected",
            "Newest", "Oldest",
        ],
        default=None,
        help=(
            "Sort order for images. Most Reactions/Most Collected "
            "are good quality proxies; Newest/Oldest are time-based."
        )
    )
    parser.add_argument(
        "--period",
        choices=["AllTime", "Year", "Month", "Week", "Day"],
        default=None,
        help="Time period for sorting"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay between image downloads in seconds"
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=None,
        help="Delay between API calls in seconds"
    )
    parser.add_argument(
        "--vault-path",
        default=None,
        help="Path to Obsidian vault (overrides config)"
    )
    parser.add_argument(
        "--nsfw",
        choices=["allow", "block", "all"],
        default=None,
        help="NSFW filter: 'allow', 'block', or 'all'"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading images (useful for testing)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Append newly-fetched images to an existing Obsidian note "
            "instead of regenerating it. Skips images that are already "
            "referenced in the note or present in the model's image "
            "folder, so reruns won't duplicate content."
        )
    )
    parser.add_argument(
        "--require-meta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Skip images that lack generation metadata. "
            "Default: on (use --no-require-meta to keep all images)."
        )
    )
    parser.add_argument(
        "--filter-begging",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Filter out images whose prompt contains 'buzz please', "
            "'give me buzz', and similar tip-begging language. "
            "Default: on (use --no-filter-begging to disable)."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )

    args = parser.parse_args()

    # Either a model arg or --interactive must be provided. Catching this
    # here (rather than via argparse) lets --interactive coexist with the
    # positional `model` being optional, so power users can still mix the
    # two if they want.
    if not args.interactive and not args.model:
        parser.error(
            "model is required (or pass --interactive / -i to be prompted)"
        )

    # --skip-download with --update would write entries pointing at
    # files that aren't on disk, leaving the note full of broken
    # embeds. Refuse the combo up front so the user can drop one or
    # the other rather than discover the damage later.
    if args.update and args.skip_download:
        print(
            "❌ --update and --skip-download cannot be combined.\n"
            "   The update flow only appends images it has actually "
            "downloaded; with --skip-download every appended embed "
            "would point at a missing file."
        )
        sys.exit(1)

    # Interactive mode runs the first-run wizard if config.yaml is
    # missing. We load config and apply CLI overrides *before* the
    # flow so its status header and pre-flight check see the effective
    # config (e.g. -i --vault-path /foo should make the flow target
    # /foo, not whatever YAML carries).
    if args.interactive:
        _require_questionary()
        if not Path(args.config).exists():
            run_config_wizard(args.config)
        config = load_config(args.config)
        _apply_cli_overrides_to_config(config, args)
        args = run_interactive_flow(config, args)
        # The flow may have flipped args.update on after the
        # skip-download default was already set; re-check the
        # mutual-exclusion guard so we fail loudly rather than later.
        if args.update and args.skip_download:
            print(
                "❌ --update and --skip-download cannot be combined."
            )
            sys.exit(1)
    else:
        config = load_config(args.config)
        _apply_cli_overrides_to_config(config, args)

    # API key resolution: the override is already in config thanks to
    # _apply_cli_overrides_to_config, so reading from config is the
    # single source of truth from here on.
    api_key = config.get("civitai", {}).get("api_key")

    # Get rate limits from args or config
    rate_limits = config.get("rate_limits", {})
    download_delay = (
        args.delay if args.delay is not None
        else rate_limits.get("download_delay", 2.0)
    )
    api_delay = (
        args.api_delay if args.api_delay is not None
        else rate_limits.get("api_delay", 1.5)
    )

    # Get defaults from args or config
    defaults = config.get("defaults", {})
    limit = (
        args.limit if args.limit is not None
        else defaults.get("image_limit", 200)
    )
    # In --update mode the whole point is to find images that weren't
    # available the last time we ran, so default to Newest/Month if the
    # user didn't pin an explicit sort. Config defaults still apply for
    # normal runs.
    if args.update:
        sort = args.sort if args.sort is not None else "Newest"
        period = args.period if args.period is not None else "Month"
    else:
        sort = (
            args.sort if args.sort is not None
            else defaults.get("sort_order", "Most Reactions")
        )
        period = (
            args.period if args.period is not None
            else defaults.get("time_period", "AllTime")
        )
    nsfw_arg = (
        args.nsfw if args.nsfw is not None
        else defaults.get("nsfw_filter", "all")
    )

    # Quality filters. Default both on — the typical use case is
    # building a curated reference library, and the failure mode for
    # "ship it broken" (a vault full of meta-less duds or buzz spam)
    # is more annoying to clean up than re-running with --no-...
    require_meta = (
        args.require_meta if args.require_meta is not None
        else _as_bool(defaults.get("require_meta"), True)
    )
    filter_begging = (
        args.filter_begging if args.filter_begging is not None
        else _as_bool(defaults.get("filter_begging"), True)
    )

    # Compile begging patterns once, up front. Built-in patterns plus
    # whatever the user has added under `defaults.begging_patterns_extra`.
    begging_patterns: List[re.Pattern[str]] = []
    if filter_begging:
        pattern_strings = list(DEFAULT_BEGGING_PATTERNS) + list(
            defaults.get("begging_patterns_extra", []) or []
        )
        begging_patterns = compile_begging_patterns(pattern_strings)

    # CLI overrides have already been folded into `config`, so the
    # helper reads everything it needs from there.
    fetcher = _build_fetcher(config)
    generator = ObsidianPageGenerator(config=config)

    try:
        # Extract model ID and optional version ID
        model_id, model_version_id = fetcher.extract_model_id(args.model)
        print(f"\n🎯 Processing CivitAI Model ID: {model_id}")
        if model_version_id:
            print(f"📌 Model Version ID: {model_version_id}")
        print()

        # Fetch model details — or reuse the payload the interactive
        # flow already pulled, so we don't hit the API twice for the
        # same model in one session.
        model_data = getattr(args, '_prefetched_model_data', None)
        if model_data is None:
            model_data = fetcher.get_model_details(model_id)
        model_name = generator.sanitize_filename(
            model_data.get("name", f"model_{model_id}")
        )

        print(f"📦 Model: {model_data.get('name')}")
        print(f"🏷️  Type: {model_data.get('type')}")

        # Convert NSFW argument to API parameter
        nsfw_param = None
        if nsfw_arg == "allow":
            nsfw_param = True
        elif nsfw_arg == "block":
            nsfw_param = False
        # "all" leaves it as None (no filter)

        # Compute target paths up front. Used regardless of mode: a
        # fresh fetch writes here, an update reads from here. Pulled
        # through the shared helper so the interactive flow's
        # pre-flight resolves identical paths.
        target_paths = compute_target_paths(
            generator, model_data, model_version_id
        )
        note_path = target_paths.note_path
        images_folder = target_paths.images_folder

        # In update mode, refuse to run without a pre-existing note —
        # otherwise the user almost certainly meant a normal run and
        # would be surprised by a fresh-looking doc with only a few
        # "Update YYYY-MM-DD" images and no original batch above them.
        existing_content: Optional[str] = None
        known_ids: set[int] = set()
        if args.update:
            expected_source = f"https://civitai.com/models/{model_id}"
            preflight = check_update_preflight(
                target_paths, expected_source
            )
            if preflight.problems:
                print(
                    "\n❌ --update pre-flight checks failed. The "
                    "following must be fixed before an update can run:"
                )
                for p in preflight.problems:
                    print(f"   • {p}")
                print(
                    "\n   Run without --update to perform a fresh "
                    "fetch, or verify the model name / version ID "
                    "matches what was used originally."
                )
                sys.exit(1)
            if preflight.warning:
                print(f"\n⚠️  {preflight.warning}")

            # preflight.existing_content is non-None whenever there
            # are no problems — the helper guarantees that contract,
            # and the assert here narrows the Optional for mypy.
            assert preflight.existing_content is not None
            existing_content = preflight.existing_content

            # If the preflight located the note via `source:` match
            # (because the user renamed it away from the computed
            # default), swap to the resolved path so the atomic
            # append below writes back to the actual file, not the
            # non-existent default. The interactive flow shows the
            # rename notice itself (earliest opportunity for the
            # user to bail) — suppress the duplicate print here.
            assert preflight.resolved_note_path is not None
            if preflight.resolved_note_path != target_paths.note_path:
                note_path = preflight.resolved_note_path
                if not args.interactive:
                    print(
                        "\nℹ️  Note located via `source:` "
                        "frontmatter (renamed from default):\n"
                        f"   {note_path}"
                    )

            md_ids = (
                ObsidianPageGenerator.extract_image_ids_from_markdown(
                    existing_content
                )
            )
            disk_ids = (
                ObsidianPageGenerator.scan_image_ids_in_folder(
                    images_folder
                )
            )
            known_ids = md_ids | disk_ids
            print(
                f"\n🔁 Update mode: found {len(md_ids)} images in note "
                f"and {len(disk_ids)} on disk ({len(known_ids)} unique)"
            )
        else:
            # Fresh-fetch safety: refuse before any API work if a
            # same-source note already exists (renamed or not) or
            # an unrelated note occupies the default path. This
            # check is cheap (one folder scan) and saves the user
            # from wasting an API quota burst + ~200 image
            # downloads on a run that we'd refuse to write anyway.
            safety_problems = check_fresh_fetch_safety(
                note_path, f"https://civitai.com/models/{model_id}"
            )
            if safety_problems:
                print("\n❌ Refusing fresh fetch:")
                for p in safety_problems:
                    print(f"\n   {p}")
                sys.exit(1)

        # Build a single accept predicate that runs INSIDE the
        # paginated fetch. This is the part that makes "--limit 200
        # images I don't already have" work: by filtering as items
        # arrive, the fetcher can keep walking pages (Most Reactions
        # 201-400, 401-600, ...) past already-known images until it
        # has collected `limit` final-quality matches. The previous
        # design ran one API call and filtered the response after
        # the fact, which silently capped the result at "200 minus
        # however many were dups", so a long-time user re-fetching
        # their best-of-AllTime model would net zero new images.
        begging_samples: List[str] = []

        def accept(img: Dict[str, Any]) -> Optional[str]:
            img_id = img.get("id")
            if img_id is None:
                return "missing_id"
            if args.update and img_id in known_ids:
                return "known"
            if require_meta and not img.get("meta"):
                return "no_meta"
            if filter_begging and begging_patterns:
                matched = detect_begging_match(img, begging_patterns)
                if matched is not None:
                    # Stash a short sample so the funnel summary
                    # can show what the filter caught, without
                    # spamming on every drop.
                    if len(begging_samples) < 5:
                        prompt = (
                            (img.get('meta') or {}).get('prompt') or ''
                        )
                        excerpt = prompt.strip().replace('\n', ' ')
                        if len(excerpt) > 80:
                            excerpt = excerpt[:77] + '...'
                        begging_samples.append(
                            f"      {img_id}: {excerpt!r}"
                        )
                    return "begging"
            return None

        # Fetch images (paginated, filtered in-flight by `accept`)
        print(f"📊 Sort order: {sort}")
        print(f"📅 Period: {period}")
        print(f"🔞 NSFW filter: {nsfw_arg}")
        fetch_result = fetcher.get_model_images(
            model_data,
            limit=limit,
            sort=sort,
            period=period,
            nsfw=nsfw_param,
            specific_version_id=model_version_id,
            api_delay=api_delay,
            accept_fn=accept,
        )
        images_data = fetch_result.images

        # Funnel summary — only emit lines that actually fired so
        # output stays readable when filters didn't drop anything.
        print(
            f"\n✅ Walked {fetch_result.pages_walked} page(s), "
            f"examined {fetch_result.candidates_seen} image candidate(s)"
        )
        drops = fetch_result.drops_by_reason
        if drops.get("video"):
            print(f"   → skipped {drops['video']} video item(s)")
        if drops.get("missing_id"):
            print(
                f"   → dropped {drops['missing_id']} item(s) with no id"
            )
        if drops.get("known"):
            print(
                f"   → dropped {drops['known']} already in note/folder"
            )
        if drops.get("no_meta"):
            print(
                f"   → dropped {drops['no_meta']} without generation "
                f"metadata"
            )
        if drops.get("begging"):
            print(
                f"   → dropped {drops['begging']} matching the "
                f"begging-spam filter"
            )
            for sample in begging_samples:
                print(sample)
            if drops["begging"] > len(begging_samples):
                print(
                    f"      ... and "
                    f"{drops['begging'] - len(begging_samples)} more"
                )
        print(f"   → {len(images_data)} image(s) will be processed")

        # Surface the page-cap warning at the top level so it's
        # impossible to miss — this is the one case where the user
        # may need to either widen sort/period or raise the cap to
        # get the full target count.
        if fetch_result.hit_page_limit and len(images_data) < limit:
            print(
                f"\n⚠️  Stopped at the per-version page cap before "
                f"reaching {limit} matches. The gallery may not have "
                "enough images matching your filters at this sort/"
                "period — try widening with --period AllTime or a "
                "different --sort."
            )

        if not images_data:
            if args.update:
                print(
                    "\n✨ Nothing new to add after filters. The "
                    "existing note is unchanged. Try widening the "
                    "search with --sort Newest --period AllTime, "
                    "raising --limit, or relaxing the filters."
                )
            else:
                print(
                    "\n✨ Nothing left after filters. Try --no-require"
                    "-meta or --no-filter-begging to broaden the set, "
                    "or raise --limit."
                )
            return

        # Create folder for images, named after the model so the user
        # can tell what's in each folder when browsing the vault.
        images_folder.mkdir(parents=True, exist_ok=True)
        print(f"📁 Images will be saved to: {images_folder}")

        # Download images
        if not args.skip_download:
            print("\n⬇️  Downloading images...")
            downloaded = 0
            for idx, image_data in enumerate(images_data, 1):
                image_url = image_data.get("url")
                if not image_url:
                    continue

                image_id = image_data.get("id", idx)
                existing = ObsidianPageGenerator.find_image_filename(
                    images_folder, image_id
                )
                if existing:
                    print(
                        f"  [{idx}/{len(images_data)}] ⏭️  Skipping "
                        f"(already exists): {existing}"
                    )
                    downloaded += 1
                    continue

                print(
                    f"  [{idx}/{len(images_data)}] 📥 Downloading: "
                    f"{image_id}"
                )
                saved = fetcher.download_image(
                    image_url,
                    images_folder,
                    image_id,
                    api_meta=image_data.get("meta"),
                )
                if saved:
                    downloaded += 1
                    print(f"      → saved as {saved.name}")

                # User-configurable rate limiting
                time.sleep(download_delay)

            print(
                f"\n✅ Downloaded {downloaded}/{len(images_data)} images"
            )

        if args.update and existing_content is not None:
            # Drop any image we tried to download but failed to land on
            # disk — otherwise the appended section would have `![[...]]`
            # entries pointing at files that aren't there. Re-scanning
            # the folder is the source of truth: if the file exists,
            # the embed will resolve; if it doesn't, the embed is dead.
            final_disk_ids = (
                ObsidianPageGenerator.scan_image_ids_in_folder(
                    images_folder
                )
            )
            before_drop = len(images_data)
            images_data = [
                img for img in images_data
                if img.get("id") in final_disk_ids
            ]
            dropped = before_drop - len(images_data)
            if dropped:
                print(
                    f"⚠️  Dropped {dropped} image(s) that failed to "
                    f"download — they will not be added to the note."
                )

            if not images_data:
                print(
                    "\n⚠️  No new images were successfully downloaded; "
                    "the note will not be modified."
                )
                return

            # Append a dated update section to the existing note rather
            # than regenerating it from scratch.
            from datetime import datetime

            update_date = datetime.now().strftime('%Y-%m-%d')
            print(
                f"\n📝 Appending update section dated {update_date}..."
            )
            update_section = generator.build_update_section(
                new_images=images_data,
                images_folder=images_folder,
                update_date=update_date
            )

            refreshed = ObsidianPageGenerator.upsert_frontmatter_field(
                existing_content, 'updated', update_date
            )
            # Separator between original body and the appended batch so
            # the new heading reads cleanly in Obsidian.
            joiner = (
                '' if refreshed.endswith('\n\n')
                else ('\n' if refreshed.endswith('\n') else '\n\n')
            )
            merged = refreshed + joiner + update_section

            # Atomic write: stage the merged content in a sibling tmp
            # file, fsync it, then rename over the original. This way
            # a crash mid-write never leaves the user's curated note
            # truncated or partially written. `Path.replace` is atomic
            # on POSIX and overwrites on Windows.
            tmp_path = note_path.with_name(
                f".{note_path.name}.update.tmp"
            )
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(merged)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.replace(note_path)
            except Exception:
                # Best-effort cleanup so we don't leave a stray tmp
                # behind for the user to wonder about.
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise

            print(f"✅ Appended {len(images_data)} new image(s) to:")
            print(f"   {note_path}")
            print(
                "\n🎉 Done! Open the page in Obsidian to review the new "
                "images at the bottom."
            )
        else:
            # Fresh-fetch safety was verified up-front (before the
            # API call), so by the time we land here we know the
            # target path is clear to write.
            print("\n📝 Generating Obsidian page...")
            page_content = generator.generate_page(
                model_data=model_data,
                images_data=images_data,
                images_folder=images_folder,
                model_name=model_name,
                specific_version_id=model_version_id
            )

            generator.save_page(
                page_content,
                note_path.name,
                model_data,
                model_version_id
            )

            print(
                "\n🎉 Done! You can now open the page in Obsidian and "
                "delete any images you don't want."
            )
            print(
                f"   Then just delete the corresponding image files "
                f"from: {images_folder}"
            )

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
