# Bug: Malformed Exif Sub-IFD next-pointer in JPEG output

**Reported from:** SagaSigil image-library reader (2026-05-16)
**Affects:** Every JPEG written by civitai-to-obsidian with EXIF UserComment metadata
**Severity:** Loud failure in strict EXIF readers (kamadak-exif, exiv2 with strict mode, some Java/Go libraries). Tolerated by piexif, Pillow's getexif, exiftool, and Civitai's own readers — hence the bug shipped invisibly.

## Symptom

A JPEG produced by civitai-to-obsidian's writer carries a fully-populated EXIF block:

- IFD0: `Software`, `Artist`, `ExifIFDPointer`
- Exif sub-IFD: `UserComment` (UNICODE-prefixed UTF-16-BE A1111 parameters)

Strict EXIF parsers reject the file with `Unexpected next IFD` (or equivalent),
discarding the entire EXIF block. The downstream consumer sees no metadata at
all — not the UserComment, not the Software, not the Artist.

In the SagaSigil reader, this surfaced as "no AI generation metadata" on every
Civitai-served JPEG that went through civitai-to-obsidian's writer.

## Root cause

The Exif sub-IFD layout is missing the 4-byte "next IFD" gap between the entry
list and the first value-by-offset payload.

### Observed file structure (105364302.jpeg, big-endian "MM")

```
Offset  Size   Field
------  -----  ------------------------------------------------------------
0x00    2      TIFF byte order ("MM")
0x02    2      TIFF magic (0x002A)
0x04    4      IFD0 offset (0x00000008)

0x08    2      IFD0 entry count (3)
0x0a    12     IFD0 entry 0: Software       (tag 0x0131, ASCII, count=37, ofs=50)
0x16    12     IFD0 entry 1: Artist         (tag 0x013b, ASCII, count=3,  inline)
0x22    12     IFD0 entry 2: ExifIFDPointer (tag 0x8769, LONG,  count=1,  ofs=87)
0x2e    4      IFD0 next-IFD pointer = 0      ✓ valid

0x32    37     Software ASCII data ("3bec7de3-8b15-4855-93a5-80a3328836ba")

0x57    2      ExifIFD entry count (1)
0x59    12     ExifIFD entry 0: UserComment (tag 0x9286, UNDEFINED, count=9170, ofs=101)
0x65    ???    ←─ STRICT READERS EXPECT 4 BYTES OF NEXT-IFD-POINTER HERE
0x65    9170   UserComment data: "UNICODE\0" + UTF-16-BE prompt payload
                ↑
                First 4 bytes "UNIC" (0x554E4943) get misread as a next-IFD
                pointer → 0x554E4943 = 1,431,193,923 → out-of-range →
                "Unexpected next IFD" error.
```

Per the TIFF 6.0 specification (the substrate EXIF builds on), every IFD —
including sub-IFDs — is followed by a 4-byte field containing the offset to
the next IFD, with `0x00000000` meaning "no next IFD." The civitai-to-obsidian
writer omits this 4-byte zero between the sub-IFD entry table and the
UserComment value, packing them back-to-back.

For the primary (IFD0) the writer does emit the zero pointer correctly — only
the Exif sub-IFD is affected.

## Why the bug was invisible

Most EXIF libraries used in the SD ecosystem (piexif, Pillow's getexif,
exiftool) treat the "next IFD" pointer on sub-IFDs as optional. They consume
the entries and stop, never attempting to read the validation bytes. Civitai's
own infrastructure reads with one of those, so round-trip looked fine in
testing.

Strict readers — kamadak-exif (Rust), some Go and C++ implementations — read
the 4 bytes following the entry list unconditionally. Any non-zero value
trips the validator.

## The fix

When writing the Exif sub-IFD, emit `\x00\x00\x00\x00` (4 bytes of zero) after
the last entry and BEFORE the value payloads start.

In other words, the offset that goes into `ExifIFDPointer` (tag 0x8769) +
`2 + N*12` must point to a 4-byte zero region, AFTER which the value-offset
payloads (UserComment data, etc.) can begin. The first UserComment value-
offset should be `(sub-IFD start) + 2 + N*12 + 4`, not `(sub-IFD start) + 2 + N*12`.

### Pseudocode for the writer

```python
def write_exif_sub_ifd(buf, entries, value_offset_base):
    # Pack the entry table.
    buf += struct.pack(">H", len(entries))        # entry count
    for tag, typ, count, value_offset in entries:
        buf += struct.pack(">HHII", tag, typ, count, value_offset)
    # CRITICAL: write the "no next IFD" sentinel before value payloads.
    buf += b"\x00\x00\x00\x00"                    # ← currently missing
    # Now append the value payloads at value_offset_base, value_offset_base+len(v0), ...
    for value_bytes in payloads:
        buf += value_bytes
```

When recomputing value-offsets, account for the +4 byte shift:

```python
sub_ifd_start = ...
entries_end = sub_ifd_start + 2 + 12 * len(entries)
next_ifd_field_end = entries_end + 4    # zero sentinel
first_value_offset = next_ifd_field_end
```

## How to verify the fix

After patching the writer, run a strict-parser sanity check on a file you've
written:

```rust
// Cargo: kamadak-exif = "0.6"
use exif::{Reader, In, Tag};
use std::{fs::File, io::BufReader};
let file = File::open("output.jpeg").unwrap();
let r = Reader::new()
    .read_from_container(&mut BufReader::new(file))
    .expect("strict EXIF read");      // must NOT error
let uc = r.get_field(Tag::UserComment, In::PRIMARY).unwrap();
println!("UserComment OK: {:?}", uc.value);
```

If it parses without `Unexpected next IFD`, the layout is now spec-compliant.

For a non-Rust check, exiftool will report the same error class if you run it
with `-validate -warning -a` against a bad file:

```
exiftool -validate -warning -a output.jpeg
# Bad file:  Warning: [minor] Unrecognized IFD entry ...
# Fixed:     (no warning)
```

## Mitigations on the reader side (already shipped in SagaSigil)

SagaSigil's reader now opts into kamadak-exif's `continue_on_error(true)` +
`distill_partial_result` recovery (commit pending in saga-sigil 2026-05-16),
which discards the bogus next-IFD validator error and retains all the
successfully-parsed entries. This means SagaSigil reads existing
already-shipped files correctly without needing a re-write. But the malformed
files will continue to cause silent failures in third-party tools (any indexer,
search tool, or LLM-pipeline that uses a strict reader), so fixing the writer
is the right long-term action.

## Sample malformed files

These three files from `Plant Milk 🌿 - Model Suite (Flax)` reproduce reliably
against any strict EXIF parser:

- `105364302.jpeg`  (UserComment: 9170 bytes)
- `108386801.jpeg`  (UserComment: 16122 bytes)
- `119749751.jpeg`  (UserComment: 1496 bytes)

The control file `101544285.jpeg` parses cleanly under strict mode, confirming
the layout difference is the issue (not the UserComment content or encoding).

## References

- TIFF 6.0 spec §2 "TIFF Structure" — IFD layout including the next-IFD pointer:
  https://www.adobe.io/open/standards/TIFF.html
- EXIF 2.32 §4.6 "Exif IFD Attribute Information" — UserComment (tag 0x9286)
- kamadak-exif `parse_child_ifd` strict validation:
  https://github.com/kamadak/exif-rs/blob/v0.6.1/src/tiff.rs#L297-L312
