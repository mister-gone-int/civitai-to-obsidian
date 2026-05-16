"""Spec-compliant EXIF writer for JPEG.

We use `piexif` to load and dump EXIF dicts because nothing else in the
Python ecosystem preserves tags as faithfully (Pillow loses thumbnails,
corrupts Unicode tags, normalises rationals lossily — verified against
a 6-bucket / 200-random-file harness; see git log for details).

`piexif` 1.1.3 has two problems we patch around here:

  1. `piexif._dump.dump` omits the 4-byte zero "next-IFD" pointer
     between an Exif / GPS / Interop sub-IFD's entry table and its
     value payloads. Per TIFF 6.0 §2 every IFD requires that pointer.
     Strict readers (kamadak-exif, exiv2 strict, several Java/Go EXIF
     libraries) reject the whole EXIF block on the malformed layout.
     Open upstream issue: hMatoba/Piexif#116 (Jan 2021), unaddressed —
     the project has had no commits since 2019.

  2. `piexif._dump._get_thumbnail` strips APPn segments from embedded
     thumbnail JPEG bytes. Saves a few bytes but is a lossy
     modification of data we promised to preserve byte-for-byte.

  3. `piexif.insert` calls `piexif._common.merge_segments`, which has
     a fragile assumption about APP segment ordering and drops the
     JFIF APP0 when both JFIF and EXIF are present (which is true for
     ~every JPEG produced by an SD pipeline). We replace this with our
     own splicer that preserves every segment byte-for-byte.

Public API:

  install_piexif_patches()    — apply (1) and (2). Idempotent.
  splice_exif_app1(jpeg_bytes, exif_payload) — solve (3).
  validate_subifd_compliance(jpeg_bytes) — verifier; returns an issue
                                           list (empty = compliant).
                                           Used as a write-time guard.
"""
from __future__ import annotations

import struct
from typing import List


# APP1 segment length field is 16 bits; the field itself is INCLUDED
# in the length count, so the payload max is (65535 - 2) bytes.
_APP1_PAYLOAD_MAX = 65533


# ---- (1) + (2) Patched piexif._dump ----------------------------------


def install_piexif_patches() -> None:
    """Install in-process patches on `piexif._dump`. Idempotent — safe
    to call multiple times. The patched functions are near-verbatim
    copies of piexif 1.1.3 with the spec fixes marked by `# PATCH:`
    comments so a diff against upstream stays auditable.
    """
    import piexif
    import piexif._dump as pd
    import numbers as _numbers

    if getattr(pd, "_civitai_patched", False):
        return

    # Constants and helpers piexif exposes through _dump (via
    # `from ._common import *`). We bind locally to keep the patched
    # functions self-contained.
    TIFF_HEADER_LENGTH = pd.TIFF_HEADER_LENGTH
    TYPES = pd.TYPES
    TAGS = pd.TAGS
    ImageIFD = pd.ImageIFD
    ExifIFD = pd.ExifIFD
    _orig_value_to_bytes = pd._value_to_bytes

    def _patched_dict_to_bytes(ifd_dict, ifd, ifd_offset):
        tag_count = len(ifd_dict)
        entry_header = struct.pack(">H", tag_count)
        # PATCH: every IFD (including Exif / GPS / Interop) needs a
        # 4-byte next-IFD pointer slot after its entry table.
        # Upstream conditionally added it only for "0th" and "1st".
        entries_length = 2 + tag_count * 12 + 4
        entries = b""
        values = b""

        for n, key in enumerate(sorted(ifd_dict)):
            if (ifd == "0th") and (
                key in (ImageIFD.ExifTag, ImageIFD.GPSTag)
            ):
                continue
            elif (ifd == "Exif") and (
                key == ExifIFD.InteroperabilityTag
            ):
                continue
            elif (ifd == "1st") and (
                key in (
                    ImageIFD.JPEGInterchangeFormat,
                    ImageIFD.JPEGInterchangeFormatLength,
                )
            ):
                continue

            raw_value = ifd_dict[key]
            key_str = struct.pack(">H", key)
            value_type = TAGS[ifd][key]["type"]
            type_str = struct.pack(">H", value_type)
            four_bytes_over = b""

            if isinstance(raw_value, _numbers.Integral) or isinstance(
                raw_value, float
            ):
                raw_value = (raw_value,)
            offset = (
                TIFF_HEADER_LENGTH + entries_length + ifd_offset
                + len(values)
            )
            try:
                length_str, value_str, four_bytes_over = (
                    _orig_value_to_bytes(raw_value, value_type, offset)
                )
            except ValueError:
                raise ValueError(
                    '"dump" got wrong type of exif value.\n'
                    + '{} in {} IFD. Got as {}.'.format(
                        key, ifd, type(ifd_dict[key])
                    )
                )
            entries += key_str + type_str + length_str + value_str
            values += four_bytes_over

        return (entry_header + entries, values)

    def _patched_dump(exif_dict_original):
        # Near-verbatim copy of piexif._dump.dump with four splice
        # points marked by `# PATCH:`. The structure is preserved to
        # make diffing against upstream easy.
        from copy import deepcopy
        exif_dict = deepcopy(exif_dict_original)

        header = b"Exif\x00\x00\x4d\x4d\x00\x2a\x00\x00\x00\x08"
        exif_is = False
        gps_is = False
        interop_is = False
        first_is = False

        zeroth_ifd = exif_dict.get("0th", {})

        if ("Exif" in exif_dict) and len(exif_dict["Exif"]):
            zeroth_ifd[ImageIFD.ExifTag] = 1
            exif_ifd = exif_dict["Exif"]
            exif_is = True
        if ("GPS" in exif_dict) and len(exif_dict["GPS"]):
            zeroth_ifd[ImageIFD.GPSTag] = 1
            gps_ifd = exif_dict["GPS"]
            gps_is = True
        if (
            ("Interop" in exif_dict)
            and len(exif_dict["Interop"])
        ):
            exif_ifd[ExifIFD.InteroperabilityTag] = 1
            interop_ifd = exif_dict["Interop"]
            interop_is = True
        if (
            ("1st" in exif_dict)
            and ("thumbnail" in exif_dict)
            and (exif_dict["thumbnail"] is not None)
        ):
            exif_dict["1st"][ImageIFD.JPEGInterchangeFormat] = 1
            exif_dict["1st"][ImageIFD.JPEGInterchangeFormatLength] = 1
            first_ifd = exif_dict["1st"]
            first_is = True

        zeroth_set = _patched_dict_to_bytes(zeroth_ifd, "0th", 0)
        zeroth_length = (
            len(zeroth_set[0])
            + exif_is * 12 + gps_is * 12 + 4
            + len(zeroth_set[1])
        )

        if exif_is:
            exif_set = _patched_dict_to_bytes(
                exif_ifd, "Exif", zeroth_length
            )
            # PATCH: +4 for the next-IFD pointer we'll splice in below.
            exif_length = (
                len(exif_set[0]) + interop_is * 12 + 4 + len(exif_set[1])
            )
        else:
            exif_bytes = b""
            exif_length = 0
        if gps_is:
            gps_set = _patched_dict_to_bytes(
                gps_ifd, "GPS", zeroth_length + exif_length
            )
            # PATCH: insert 4 zero bytes between entries and values.
            gps_bytes = (
                gps_set[0] + b"\x00\x00\x00\x00" + gps_set[1]
            )
            gps_length = len(gps_bytes)
        else:
            gps_bytes = b""
            gps_length = 0
        if interop_is:
            offset = zeroth_length + exif_length + gps_length
            interop_set = _patched_dict_to_bytes(
                interop_ifd, "Interop", offset
            )
            # PATCH: insert 4 zero bytes between entries and values.
            interop_bytes = (
                interop_set[0] + b"\x00\x00\x00\x00" + interop_set[1]
            )
            interop_length = len(interop_bytes)
        else:
            interop_bytes = b""
            interop_length = 0
        if first_is:
            offset = (
                zeroth_length + exif_length + gps_length + interop_length
            )
            first_set = _patched_dict_to_bytes(first_ifd, "1st", offset)
            thumbnail = pd._get_thumbnail(exif_dict["thumbnail"])
            if len(thumbnail) > 64000:
                raise ValueError(
                    "Given thumbnail is too large. max 64kB"
                )
        else:
            first_bytes = b""

        if exif_is:
            pointer_value = TIFF_HEADER_LENGTH + zeroth_length
            pointer_str = struct.pack(">I", pointer_value)
            key = ImageIFD.ExifTag
            key_str = struct.pack(">H", key)
            type_str = struct.pack(">H", TYPES.Long)
            length_str = struct.pack(">I", 1)
            exif_pointer = (
                key_str + type_str + length_str + pointer_str
            )
        else:
            exif_pointer = b""
        if gps_is:
            pointer_value = (
                TIFF_HEADER_LENGTH + zeroth_length + exif_length
            )
            pointer_str = struct.pack(">I", pointer_value)
            key = ImageIFD.GPSTag
            key_str = struct.pack(">H", key)
            type_str = struct.pack(">H", TYPES.Long)
            length_str = struct.pack(">I", 1)
            gps_pointer = (
                key_str + type_str + length_str + pointer_str
            )
        else:
            gps_pointer = b""
        if interop_is:
            pointer_value = (
                TIFF_HEADER_LENGTH + zeroth_length
                + exif_length + gps_length
            )
            pointer_str = struct.pack(">I", pointer_value)
            key = ExifIFD.InteroperabilityTag
            key_str = struct.pack(">H", key)
            type_str = struct.pack(">H", TYPES.Long)
            length_str = struct.pack(">I", 1)
            interop_pointer = (
                key_str + type_str + length_str + pointer_str
            )
        else:
            interop_pointer = b""
        if first_is:
            pointer_value = (
                TIFF_HEADER_LENGTH + zeroth_length
                + exif_length + gps_length + interop_length
            )
            first_ifd_pointer = struct.pack(">L", pointer_value)
            thumbnail_pointer = (
                pointer_value + len(first_set[0]) + 24 + 4
                + len(first_set[1])
            )
            thumbnail_p_bytes = (
                b"\x02\x01\x00\x04\x00\x00\x00\x01"
                + struct.pack(">L", thumbnail_pointer)
            )
            thumbnail_length_bytes = (
                b"\x02\x02\x00\x04\x00\x00\x00\x01"
                + struct.pack(">L", len(thumbnail))
            )
            first_bytes = (
                first_set[0] + thumbnail_p_bytes
                + thumbnail_length_bytes
                + b"\x00\x00\x00\x00" + first_set[1] + thumbnail
            )
        else:
            first_ifd_pointer = b"\x00\x00\x00\x00"

        zeroth_bytes = (
            zeroth_set[0] + exif_pointer + gps_pointer
            + first_ifd_pointer + zeroth_set[1]
        )
        if exif_is:
            # PATCH: 4 zero bytes between sub-IFD entries and values.
            exif_bytes = (
                exif_set[0] + interop_pointer
                + b"\x00\x00\x00\x00" + exif_set[1]
            )

        return (
            header + zeroth_bytes + exif_bytes + gps_bytes
            + interop_bytes + first_bytes
        )

    # Preserve embedded-thumbnail bytes verbatim. piexif's default
    # strips APPn segments from the thumbnail JPEG — visually harmless
    # but a real byte-level mutation. We promised callers byte-for-byte
    # preservation, so we override.
    def _patched_get_thumbnail(jpeg):
        return jpeg

    pd._dict_to_bytes = _patched_dict_to_bytes
    pd._get_thumbnail = _patched_get_thumbnail
    pd.dump = _patched_dump
    piexif.dump = _patched_dump
    pd._civitai_patched = True


# ---- (3) Byte-perfect APP1-EXIF splicer ------------------------------


class ExifSpliceError(ValueError):
    """Raised when a JPEG can't be safely modified — bad input, EXIF
    payload too large for an APP1 segment, etc. The original file is
    NEVER modified; the caller can treat this as a refusal."""


_JPEG_SOI = b"\xff\xd8"
_JPEG_SOS = 0xda


def splice_exif_app1(
    jpeg_bytes: bytes, new_exif_payload: bytes
) -> bytes:
    """Return `jpeg_bytes` with its EXIF APP1 segment replaced by
    `new_exif_payload` (which must start with `b"Exif\\x00\\x00"`).

    Guarantees:
      - Every non-EXIF segment is preserved byte-for-byte (JFIF APP0,
        APP1-XMP, APP2-ICC, COM, DQT, DHT, SOF, SOS header, entropy-
        coded scan data, EOI).
      - If no existing EXIF APP1 is present, the new APP1 is inserted
        immediately after the JFIF APP0 (or after SOI if there is no
        JFIF), matching the conventional segment order.
      - If multiple EXIF APP1 segments exist (malformed input), only
        the first is kept and replaced; subsequent ones are dropped to
        avoid leaving stale conflicting data downstream.

    Raises ExifSpliceError on bad input or oversize payload.
    """
    if not new_exif_payload.startswith(b"Exif\x00\x00"):
        raise ExifSpliceError(
            "EXIF payload must start with b'Exif\\x00\\x00'"
        )
    if len(new_exif_payload) > _APP1_PAYLOAD_MAX:
        raise ExifSpliceError(
            f"EXIF payload of {len(new_exif_payload)} bytes exceeds "
            f"APP1 segment limit of {_APP1_PAYLOAD_MAX}"
        )
    if jpeg_bytes[:2] != _JPEG_SOI:
        raise ExifSpliceError("input is not a JPEG (missing SOI)")

    new_app1 = (
        b"\xff\xe1"
        + struct.pack(">H", len(new_exif_payload) + 2)
        + new_exif_payload
    )

    # Walk markers up to SOS. Every segment we encounter is either:
    #   - emitted verbatim into `out`
    #   - the EXIF-APP1 we replace (first one only — subsequent
    #     EXIF-APP1s are dropped as conflicting)
    out = bytearray(_JPEG_SOI)
    exif_handled = False
    last_jfif_end = None  # position in `out` immediately after JFIF
    i = 2
    while i < len(jpeg_bytes):
        if jpeg_bytes[i] != 0xff:
            raise ExifSpliceError(
                f"bad marker prefix at offset {i:#x}: {jpeg_bytes[i]:#x}"
            )
        marker = jpeg_bytes[i + 1]
        if marker == _JPEG_SOS:
            # Hit Start-of-Scan. Everything from here to EOF is the
            # entropy-coded data (including restart markers) and EOI —
            # emit byte-for-byte.
            if not exif_handled:
                # No existing EXIF found; splice the new APP1 into
                # the conventional slot: right after JFIF if present,
                # else immediately after SOI. Because we've only
                # copied verbatim into `out` so far, positions in
                # `out` and `jpeg_bytes` are identical for the
                # consumed prefix — so a split-and-rejoin is safe.
                insertion = (
                    last_jfif_end if last_jfif_end is not None else 2
                )
                head = bytes(out[:insertion])
                tail = jpeg_bytes[insertion:]
                return head + new_app1 + tail
            out.extend(jpeg_bytes[i:])
            return bytes(out)

        length = struct.unpack(">H", jpeg_bytes[i + 2:i + 4])[0]
        seg_end = i + 2 + length
        seg_payload = jpeg_bytes[i + 4:seg_end]

        if marker == 0xe0 and seg_payload[:5] == b"JFIF\x00":
            out.extend(jpeg_bytes[i:seg_end])
            last_jfif_end = len(out)
        elif (
            marker == 0xe1
            and seg_payload[:6] == b"Exif\x00\x00"
        ):
            if not exif_handled:
                out.extend(new_app1)
                exif_handled = True
            # else: silently drop duplicate EXIF APP1
        else:
            out.extend(jpeg_bytes[i:seg_end])
        i = seg_end

    raise ExifSpliceError("no SOS marker found in JPEG")


# ---- (verifier) Strict-mode sub-IFD compliance check -----------------


def validate_subifd_compliance(jpeg_bytes: bytes) -> List[str]:
    """Walk the EXIF APP1 payload and confirm Exif / GPS / Interop
    sub-IFDs all have the spec-mandated 4-byte zero "next-IFD" pointer
    between their entry table and value payloads.

    Returns a list of human-readable issues; empty list means the
    output passes strict-EXIF parsers (kamadak-exif, exiv2 strict).
    Designed for use as a write-time guard.
    """
    issues: List[str] = []
    exif_payload = None
    i = 2
    if jpeg_bytes[:2] != _JPEG_SOI:
        return ["not a JPEG"]
    while i < len(jpeg_bytes):
        if jpeg_bytes[i] != 0xff:
            return [f"bad marker prefix at offset {i:#x}"]
        marker = jpeg_bytes[i + 1]
        if marker == _JPEG_SOS:
            break
        length = struct.unpack(">H", jpeg_bytes[i + 2:i + 4])[0]
        seg_end = i + 2 + length
        seg_payload = jpeg_bytes[i + 4:seg_end]
        if (
            marker == 0xe1
            and seg_payload[:6] == b"Exif\x00\x00"
        ):
            exif_payload = seg_payload[6:]
            break
        i = seg_end

    if exif_payload is None:
        # No EXIF — vacuously compliant.
        return issues

    bo = exif_payload[:2]
    if bo == b"MM":
        E = ">"
    elif bo == b"II":
        E = "<"
    else:
        return [f"bad TIFF byte-order marker: {bo!r}"]
    ifd0_off = struct.unpack(E + "I", exif_payload[4:8])[0]

    visited: set[int] = set()
    # (ifd_offset, is_sub_ifd, label)
    todo = [(ifd0_off, False, "IFD0")]
    while todo:
        ifd_off, is_sub, label = todo.pop()
        if ifd_off in visited or ifd_off + 2 > len(exif_payload):
            continue
        visited.add(ifd_off)
        try:
            n = struct.unpack(
                E + "H", exif_payload[ifd_off:ifd_off + 2]
            )[0]
        except struct.error:
            issues.append(f"{label}: truncated entry count")
            continue
        entries_end = ifd_off + 2 + n * 12
        if entries_end + 4 > len(exif_payload):
            issues.append(f"{label}: truncated entries")
            continue
        if is_sub:
            next_bytes = exif_payload[entries_end:entries_end + 4]
            if next_bytes != b"\x00\x00\x00\x00":
                issues.append(
                    f"{label} sub-IFD has non-zero bytes at "
                    f"entries_end ({next_bytes.hex()}) — strict-EXIF "
                    f"readers will reject this layout"
                )
        for k in range(n):
            pos = ifd_off + 2 + k * 12
            tag, _typ, _count, val = struct.unpack(
                E + "HHII", exif_payload[pos:pos + 12]
            )
            if tag == 0x8769:
                todo.append((val, True, "Exif"))
            elif tag == 0x8825:
                todo.append((val, True, "GPS"))
            elif tag == 0xA005 and label == "Exif":
                todo.append((val, True, "Interop"))
    return issues
