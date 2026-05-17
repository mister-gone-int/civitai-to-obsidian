"""Surgical PNG chunk-CRC repair.

CivitAI's CDN occasionally serves PNG files where one or more chunk
CRCs are wrong even though the chunk content itself is intact (most
often the tEXt chunk holding generation parameters). PIL refuses to
open such files with `UnidentifiedImageError`, even though the pixel
data and metadata are perfectly recoverable — only the 4-byte CRC
trailer is incorrect.

This module repairs that specific defect in place: walk chunks,
recompute CRCs, rewrite only the CRC trailer bytes. Chunk content
and ordering are never modified, so the repaired bytes round-trip
through any PNG reader (PIL, browsers, image-library apps) and the
embedded metadata is byte-identical to what the user downloaded.

Safety properties (verified by `_assert_byte_equal_except_crcs`):
  - The repaired output and the original differ ONLY in CRC fields.
  - No chunk is added, removed, reordered, or content-modified.
  - The PNG signature, chunk lengths, chunk types are preserved.
  - If the file is malformed beyond bad CRCs (truncated header,
    missing IEND, unparseable length), the original bytes are
    returned unchanged with a structural-error note — never a
    partial or guess-based repair.

Use `repair_png_bytes` at download time (post-fetch / pre-save) or
on existing files via `repair_png_file` (atomic write + PIL verify).
"""

from __future__ import annotations

import os
import struct
import tempfile
import zlib
from io import BytesIO
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

# PNG signature: bytes that every valid PNG file begins with. We
# refuse to touch anything that doesn't lead with this — better to
# leave a JPEG or junk file alone than to "repair" non-PNG bytes.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class _Chunk(NamedTuple):
    """Parsed view of one PNG chunk's position in the file."""
    offset: int          # byte offset of the chunk's 4-byte length
    length: int          # data field length (excludes type / CRC)
    ctype_bytes: bytes   # 4 raw type bytes (e.g. b"tEXt")
    data_start: int      # byte offset of the chunk data
    data_end: int        # one past last byte of chunk data
    crc_stored: int      # CRC as currently written in the file
    crc_actual: int      # CRC recomputed from type + data


class RepairResult(NamedTuple):
    """Outcome of a repair attempt.

    - `data`: the (possibly-repaired) bytes. Always returned, even
      when no repair was performed, so callers don't need a separate
      "did we change anything" branch — just use the bytes.
    - `notes`: human-readable summary of what was done. Empty when
      no repairs were applied AND no anomalies were found.
    - `structural_error`: set when the file is malformed beyond CRC
      bit-rot. When non-None, `data` is the ORIGINAL bytes (we never
      attempt partial repair on unrecognised structure).
    """
    data: bytes
    notes: List[str]
    structural_error: Optional[str]


def repair_png_bytes(data: bytes) -> RepairResult:
    """Recompute and rewrite bad chunk CRCs in a PNG byte sequence.

    Returns the (possibly-repaired) bytes alongside a notes list and
    an optional structural-error message. When `structural_error` is
    set, the original bytes are returned unchanged — the file is too
    damaged for a safe CRC-only repair.
    """
    # Guard against non-bytes input WITHOUT calling bytes(data) — for
    # str/int/None that would raise or fabricate a zero-filled buffer
    # of `data` bytes. Return an empty payload with a clear note; the
    # caller can decide what to do.
    if not isinstance(data, (bytes, bytearray)):
        return RepairResult(b"", [], "not bytes")

    if not data.startswith(_PNG_SIGNATURE):
        return RepairResult(bytes(data),
                            [],
                            "missing PNG signature")

    chunks, struct_err = _parse_chunks(data)
    if struct_err is not None:
        return RepairResult(bytes(data), [], struct_err)
    if not chunks:
        return RepairResult(bytes(data), [], "no chunks found")
    if chunks[0].ctype_bytes != b"IHDR":
        return RepairResult(bytes(data), [], "first chunk is not IHDR")
    if chunks[-1].ctype_bytes != b"IEND":
        return RepairResult(bytes(data), [], "last chunk is not IEND")

    # Collect chunks that need a CRC fix. We don't mutate the buffer
    # until we know we have a clean repair plan — keeps the function
    # naturally atomic in memory (caller sees either the original or
    # a fully-repaired version, never a half-rewritten one).
    bad = [c for c in chunks if c.crc_stored != c.crc_actual]
    if not bad:
        return RepairResult(bytes(data), [], None)

    # Rewrite only the 4-byte CRC trailer of each bad chunk. Using a
    # bytearray for in-place edits — chunk content positions don't
    # shift because we never change lengths, just CRC bytes.
    repaired = bytearray(data)
    notes: List[str] = []
    for c in bad:
        crc_offset = c.data_end  # 4 bytes of CRC live here
        struct.pack_into(">I", repaired, crc_offset, c.crc_actual)
        notes.append(
            f"fixed CRC on {c.ctype_bytes.decode('ascii', 'replace')} "
            f"chunk @ {c.offset} (len={c.length}): "
            f"0x{c.crc_stored:08x} → 0x{c.crc_actual:08x}"
        )

    out = bytes(repaired)
    _assert_byte_equal_except_crcs(data, out, bad)
    return RepairResult(out, notes, None)


def repair_png_file(
    path: Path,
    verify_with_pil: bool = True,
) -> RepairResult:
    """Read, repair, and (atomically) rewrite a PNG file on disk.

    The file is only replaced when:
      1. At least one CRC was bad (no churn when the file is fine).
      2. The repaired bytes pass a PIL.Image.open() + .verify() check
         (when `verify_with_pil` is True).

    On structural errors or verification failure, the file is left
    untouched and the failure is returned in `structural_error`.

    Returns the RepairResult as `repair_png_bytes`, with an extra
    note appended when the file is actually written to disk.
    """
    try:
        original = path.read_bytes()
    except OSError as exc:
        return RepairResult(b"", [], f"read failed: {exc}")

    result = repair_png_bytes(original)
    if result.structural_error or not result.notes:
        return result

    if verify_with_pil:
        try:
            # Lazy import: callers that don't pass verify_with_pil
            # (e.g. headless utilities) don't pay the PIL import cost.
            from PIL import Image, UnidentifiedImageError
            with Image.open(BytesIO(result.data)) as im:
                im.verify()
        except (UnidentifiedImageError, OSError,
                ValueError, SyntaxError) as exc:
            # The documented set PIL raises during open()/verify().
            # We intentionally don't catch `Exception` — a programming
            # bug (AttributeError, TypeError) should surface loudly,
            # not be mislabelled as "PIL verify failed".
            return RepairResult(
                original,
                [],
                f"post-repair PIL verify failed: {exc}",
            )

    # Atomic write: unique tempfile in the same directory (so the
    # final os.replace is a same-filesystem rename), then atomic
    # rename onto the target. NamedTemporaryFile with delete=False +
    # a unique suffix avoids two failure modes a deterministic tmp
    # name suffers from: (a) concurrent invocations clobbering each
    # other, and (b) a leftover from a previously-crashed run getting
    # silently overwritten.
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
            prefix=path.name + ".",
            suffix=".repair-tmp",
        ) as tf:
            tf.write(result.data)
            tmp_path = Path(tf.name)
        os.replace(tmp_path, path)
    except OSError as exc:
        # Best-effort cleanup; don't shadow the original error.
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return RepairResult(original, [], f"write failed: {exc}")

    return RepairResult(
        result.data,
        result.notes + [f"wrote repaired file: {path.name}"],
        None,
    )


# ---- internals -----------------------------------------------------

def _parse_chunks(
    data: bytes,
) -> Tuple[List[_Chunk], Optional[str]]:
    """Walk PNG chunks starting after the signature.

    Returns (chunks, error). `error` is set on any structural anomaly
    (truncated chunk header, length overflow, missing IEND); the
    chunks list collected so far is returned alongside but should
    not be used for repair when error is non-None.
    """
    chunks: List[_Chunk] = []
    pos = len(_PNG_SIGNATURE)
    saw_iend = False
    while pos < len(data):
        if pos + 8 > len(data):
            return chunks, f"truncated chunk header @ {pos}"
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        data_start = pos + 8
        data_end = data_start + length
        if data_end + 4 > len(data):
            return (
                chunks,
                f"chunk {ctype!r} @ {pos} truncated "
                f"(len={length}, file remains {len(data) - data_start})",
            )
        # Validate chunk type bytes are ASCII letters per PNG spec —
        # otherwise the length we just read is almost certainly wrong
        # and we'd march off into garbage.
        if not all(0x41 <= b <= 0x7A and (b <= 0x5A or b >= 0x61)
                   for b in ctype):
            return chunks, f"non-ASCII chunk type @ {pos}: {ctype!r}"
        crc_stored = struct.unpack(">I", data[data_end:data_end + 4])[0]
        crc_actual = zlib.crc32(data[pos + 4:data_end]) & 0xFFFFFFFF
        chunks.append(_Chunk(
            offset=pos,
            length=length,
            ctype_bytes=ctype,
            data_start=data_start,
            data_end=data_end,
            crc_stored=crc_stored,
            crc_actual=crc_actual,
        ))
        pos = data_end + 4
        if ctype == b"IEND":
            saw_iend = True
            break
    if not saw_iend:
        return chunks, "no IEND chunk before end of file"
    return chunks, None


def _assert_byte_equal_except_crcs(
    before: bytes, after: bytes, repaired: List[_Chunk],
) -> None:
    """Defensive check: the only differences between `before` and
    `after` must be inside the 4-byte CRC trailers of the chunks we
    intentionally repaired. Anything else is a bug in this module.

    Raises AssertionError loudly so a development regression can't
    silently ship a corrupted file. Cost is one byte-string compare
    per chunk — negligible vs the safety guarantee.
    """
    if len(before) != len(after):
        raise AssertionError(
            f"length changed: {len(before)} → {len(after)}"
        )
    crc_ranges = {(c.data_end, c.data_end + 4) for c in repaired}
    # Compare in slices, skipping the known-changed CRC ranges.
    last = 0
    for start, end in sorted(crc_ranges):
        if before[last:start] != after[last:start]:
            raise AssertionError(
                f"non-CRC bytes changed in range [{last}:{start})"
            )
        last = end
    if before[last:] != after[last:]:
        raise AssertionError(
            f"non-CRC bytes changed in range [{last}:end)"
        )
