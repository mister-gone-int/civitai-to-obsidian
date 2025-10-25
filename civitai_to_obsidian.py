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

                if items:
                    all_images.extend(items)
                    print(f"  ✓ Got {len(items)} images")
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

    def download_image(self, url: str, output_path: Path) -> bool:
        """Download an image from URL to local path"""
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False


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

            lines.append(f"#### Image {idx}\n")

            # Image embed - using relative path from vault root
            image_filename = f"{image_id}.jpeg"
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
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )

    args = parser.parse_args()

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

        # Filter out images without metadata if user wants quality data
        images_with_meta = [
            img for img in images_data if img.get('meta')
        ]
        print(
            f"\n✅ Fetched {len(images_data)} images "
            f"({len(images_with_meta)} with generation metadata)"
        )

        # Create folder for images using model_id and version_id
        if model_version_id:
            folder_name = f"{model_id}_v{model_version_id}"
        else:
            folder_name = str(model_id)
        images_folder = generator.media_folder / folder_name
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
                image_filename = f"{image_id}.jpeg"
                image_path = images_folder / image_filename

                if image_path.exists():
                    print(
                        f"  [{idx}/{len(images_data)}] ⏭️  Skipping "
                        f"(already exists): {image_filename}"
                    )
                    downloaded += 1
                    continue

                print(
                    f"  [{idx}/{len(images_data)}] 📥 Downloading: "
                    f"{image_filename}"
                )
                if fetcher.download_image(image_url, image_path):
                    downloaded += 1

                # User-configurable rate limiting
                time.sleep(download_delay)

            print(
                f"\n✅ Downloaded {downloaded}/{len(images_data)} images"
            )

        # Generate Obsidian page
        print("\n📝 Generating Obsidian page...")
        page_content = generator.generate_page(
            model_data=model_data,
            images_data=images_data,
            images_folder=images_folder,
            model_name=model_name,
            specific_version_id=model_version_id
        )

        # Save page with formatted title
        formatted_title = generator.format_title(
            model_data,
            model_version_id
        )
        page_filename = f"{formatted_title}.md"
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
            f"   Then just delete the corresponding image files from: "
            f"{images_folder}"
        )

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
