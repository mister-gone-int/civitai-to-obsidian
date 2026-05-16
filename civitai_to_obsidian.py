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
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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

    def extract_model_id(
        self,
        url_or_id: str
    ) -> tuple[int, Optional[int]]:
        """Extract model ID and optional modelVersionId from URL or ID

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

    def get_model_images(
        self,
        model_data: Dict[str, Any],
        limit: int = 200,
        sort: str = "Most Reactions",
        period: str = "AllTime",
        nsfw: Optional[bool] = None,
        specific_version_id: Optional[int] = None,
        api_delay: float = 1.5
    ) -> List[Dict[str, Any]]:
        """Fetch images for a model using modelVersionId

        Args:
            model_data: Model data from API
            limit: Max images to fetch
            sort: Sort order
            period: Time period for sorting
            nsfw: Filter NSFW content (True=allow, False=SFW, None=all)
            specific_version_id: If provided, only fetch images for this
                                version
            api_delay: Delay between API calls
        """
        # Get all model versions
        versions = model_data.get("modelVersions", [])
        if not versions:
            print("No model versions found")
            return []

        all_images: List[Dict[str, Any]] = []

        # If specific version ID is provided, filter to that version only
        if specific_version_id:
            versions = [
                v for v in versions if v.get("id") == specific_version_id
            ]
            if not versions:
                print(
                    f"⚠️  Warning: Model version ID {specific_version_id} "
                    "not found in this model"
                )
                return []
            print(
                f"🎯 Filtering to specific version ID: "
                f"{specific_version_id}"
            )

        # Fetch images for each version until we hit the limit
        for version in versions:
            if len(all_images) >= limit:
                break

            version_id = version.get("id")
            version_name = version.get("name", "Unknown")

            print(
                f"Fetching images for version: {version_name} "
                f"(ID: {version_id})..."
            )

            url = f"{self.base_url}/images"
            params: Dict[str, Any] = {
                "modelVersionId": version_id,
                "limit": min(limit - len(all_images), 200),
                "sort": sort,
                "period": period
            }

            # Add NSFW filter if specified
            if nsfw is not None:
                params["nsfw"] = str(nsfw).lower()

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()
                items = data.get("items", [])

                # Filter out videos — CivitAI hosts MP4 clips alongside
                # images, but we only embed images in Obsidian.
                image_items = [
                    i for i in items if i.get("type", "image") == "image"
                ]
                skipped = len(items) - len(image_items)

                if image_items:
                    all_images.extend(image_items)
                    suffix = (
                        f" (skipped {skipped} video(s))" if skipped else ""
                    )
                    print(f"  ✓ Got {len(image_items)} images{suffix}")
                elif skipped:
                    print(f"  All {skipped} items were videos — skipped")
                else:
                    print("  No images for this version")

            except Exception as e:
                print(
                    f"  Error fetching images for version "
                    f"{version_id}: {e}"
                )
                continue

            # API rate limiting
            time.sleep(api_delay)

        return all_images[:limit]

    def download_image(
        self,
        url: str,
        output_dir: Path,
        image_id: Any
    ) -> Optional[Path]:
        """Download an image and save with extension inferred from bytes.

        The CivitAI CDN serves files whose URL extension doesn't always
        match the actual content (a URL ending in .jpeg may be a PNG),
        so we sniff the magic bytes and pick the extension ourselves.
        Returns the final saved Path, or None if the download failed or
        the content wasn't a supported image format (e.g. video).
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

            output_path = output_dir / f"{image_id}.{ext}"
            with open(output_path, 'wb') as f:
                f.write(content)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch CivitAI model images and create Obsidian docs"
    )
    parser.add_argument(
        "model",
        help="CivitAI model URL or ID"
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
        choices=["Most Reactions", "Newest", "Most Comments"],
        default=None,
        help="Sort order for images"
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

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.vault_path:
        config.setdefault("obsidian", {})["vault_path"] = args.vault_path

    # Get API key from args or config
    api_key = args.api_key or config.get("civitai", {}).get("api_key")

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
        else defaults.get("require_meta", True)
    )
    filter_begging = (
        args.filter_begging if args.filter_begging is not None
        else defaults.get("filter_begging", True)
    )

    # Compile begging patterns once, up front. Built-in patterns plus
    # whatever the user has added under `defaults.begging_patterns_extra`.
    begging_patterns: List[re.Pattern[str]] = []
    if filter_begging:
        pattern_strings = list(DEFAULT_BEGGING_PATTERNS) + list(
            defaults.get("begging_patterns_extra", []) or []
        )
        begging_patterns = compile_begging_patterns(pattern_strings)

    # Initialize
    civitai_config = config.get("civitai", {})
    fetcher = CivitAIFetcher(
        api_key=api_key,
        base_url=civitai_config.get(
            "base_url",
            "https://civitai.com/api/v1"
        ),
        max_retries=rate_limits.get("max_retries", 3),
        backoff_factor=rate_limits.get("backoff_factor", 1)
    )
    generator = ObsidianPageGenerator(config=config)

    try:
        # Extract model ID and optional version ID
        model_id, model_version_id = fetcher.extract_model_id(args.model)
        print(f"\n🎯 Processing CivitAI Model ID: {model_id}")
        if model_version_id:
            print(f"📌 Model Version ID: {model_version_id}")
        print()

        # Fetch model details
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

        # Compute target note path up front. In --update mode we need
        # to read the existing file before fetching, so we always
        # resolve the path here regardless of mode.
        formatted_title = generator.format_title(
            model_data,
            model_version_id
        )
        page_filename = f"{formatted_title}.md"
        note_dir = generator.get_note_directory(
            model_data, model_version_id
        )
        note_path = note_dir / page_filename

        folder_name = ObsidianPageGenerator.build_image_folder_name(
            model_data, model_version_id
        )
        images_folder = generator.media_folder / folder_name

        # In update mode, refuse to run without a pre-existing note —
        # otherwise the user almost certainly meant a normal run and
        # would be surprised by a fresh-looking doc with only a few
        # "Update YYYY-MM-DD" images and no original batch above them.
        existing_content: Optional[str] = None
        known_ids: set[int] = set()
        if args.update:
            # Pre-flight checks. Both the note and the image folder
            # must already exist; bail out with a precise diagnostic if
            # either is missing so the user knows which side to fix.
            problems: List[str] = []
            if not note_path.exists():
                problems.append(
                    f"Obsidian note not found at:\n      {note_path}"
                )
            elif not note_path.is_file():
                problems.append(
                    f"Path exists but is not a regular file:\n"
                    f"      {note_path}"
                )
            if not images_folder.exists():
                problems.append(
                    f"Image folder not found at:\n      "
                    f"{images_folder}"
                )
            elif not images_folder.is_dir():
                problems.append(
                    f"Image folder path exists but is not a "
                    f"directory:\n      {images_folder}"
                )

            if problems:
                print(
                    "\n❌ --update pre-flight checks failed. The "
                    "following must exist before an update can run:"
                )
                for p in problems:
                    print(f"   • {p}")
                print(
                    "\n   Run without --update to perform a fresh "
                    "fetch, or verify the model name / version ID "
                    "matches what was used originally."
                )
                sys.exit(1)

            existing_content = note_path.read_text(encoding='utf-8')

            # Verify the note we found actually corresponds to this
            # model. Two different models can produce the same title
            # (e.g. both LoRAs named "Style Test" on SDXL), and we'd
            # otherwise dedupe against — and append to — the wrong
            # note. The `source:` line in our generated frontmatter is
            # the authoritative pointer back to the CivitAI model.
            existing_source = (
                ObsidianPageGenerator.extract_frontmatter_field(
                    existing_content, 'source'
                )
            )
            expected_source = (
                f"https://civitai.com/models/{model_id}"
            )
            if existing_source is None:
                print(
                    "\n⚠️  Existing note has no `source:` field in its "
                    "frontmatter — skipping model-match verification. "
                    "If this note wasn't generated by this script, "
                    "double-check the path is right before proceeding."
                )
            elif existing_source.rstrip('/') != expected_source:
                print(
                    "\n❌ The existing note at this path belongs to a "
                    "different model.\n"
                    f"   Expected source: {expected_source}\n"
                    f"   Note's source:   {existing_source}\n"
                    "   Refusing to update — appending here would "
                    "corrupt that other note."
                )
                sys.exit(1)

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

        # Fetch images
        print(f"📊 Sort order: {sort}")
        print(f"📅 Period: {period}")
        print(f"🔞 NSFW filter: {nsfw_arg}")
        images_data = fetcher.get_model_images(
            model_data,
            limit=limit,
            sort=sort,
            period=period,
            nsfw=nsfw_param,
            specific_version_id=model_version_id,
            api_delay=api_delay
        )

        api_count = len(images_data)
        print(f"\n✅ Fetched {api_count} image(s) from API")

        # Quality filters are applied in priority order: first drop
        # already-processed images (cheapest, update-mode only), then
        # require generation metadata (needed for downstream checks),
        # then run the begging filter (most expensive — regex per
        # prompt). Reporting is consolidated at the bottom so the user
        # sees a single coherent funnel.
        dropped_known = 0
        if args.update:
            before = len(images_data)
            images_data = [
                img for img in images_data
                if img.get("id") not in known_ids
            ]
            dropped_known = before - len(images_data)

        dropped_meta = 0
        if require_meta:
            before = len(images_data)
            images_data = [
                img for img in images_data if img.get('meta')
            ]
            dropped_meta = before - len(images_data)

        dropped_begging = 0
        begging_samples: List[str] = []
        if filter_begging and begging_patterns:
            kept: List[Dict[str, Any]] = []
            for img in images_data:
                matched = detect_begging_match(img, begging_patterns)
                if matched is None:
                    kept.append(img)
                    continue
                dropped_begging += 1
                # Stash a short, human-readable sample for the summary
                # so the user can verify the filter is doing the right
                # thing without scrolling through a wall of output.
                if len(begging_samples) < 5:
                    prompt = (img.get('meta') or {}).get('prompt') or ''
                    excerpt = prompt.strip().replace('\n', ' ')
                    if len(excerpt) > 80:
                        excerpt = excerpt[:77] + '...'
                    begging_samples.append(
                        f"      {img.get('id')}: {excerpt!r}"
                    )
            images_data = kept

        # Funnel summary — only print the lines that actually fired so
        # the output stays tight when filters didn't drop anything.
        if dropped_known:
            print(
                f"   → dropped {dropped_known} already in note/folder"
            )
        if dropped_meta:
            print(
                f"   → dropped {dropped_meta} without generation "
                f"metadata"
            )
        if dropped_begging:
            print(
                f"   → dropped {dropped_begging} matching the "
                f"begging-spam filter"
            )
            for sample in begging_samples:
                print(sample)
            if dropped_begging > len(begging_samples):
                print(
                    f"      ... and "
                    f"{dropped_begging - len(begging_samples)} more"
                )
        print(f"   → {len(images_data)} image(s) will be processed")

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
                    image_url, images_folder, image_id
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
            import os
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
            # Generate Obsidian page
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
                page_filename,
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
