# CivitAI to Obsidian Image Library Builder

When you pull down a new model or LoRA from CivitAI, it's helpful to keep example images and their generation parameters around as a reference. You can see what the model is capable of, what styles it produces, and which prompts and settings other people have had success with. Obsidian works well for this kind of reference library, but building it by hand for every model gets tedious fast.

This script does the work for you. Give it a model URL or ID and it will fetch example images from CivitAI, download them into your vault's media folder, pull out all the generation metadata, and write an Obsidian markdown page that embeds the images alongside their prompts and parameters.

<img width="2157" height="1981" alt="image" src="https://github.com/user-attachments/assets/a12cd072-7d45-48b5-83db-be7f2ce2ccde" />

## What it does

- Fetches images per model version, which avoids the timeouts you hit when querying across all versions at once
- Downloads images to your Obsidian vault's media folder, organized by model
- Extracts generation parameters: prompts, sampler, CFG scale, steps, seed, size, LoRAs and weights, and so on
- Writes an Obsidian markdown page with YAML frontmatter (tags, author, source link, creator, downloads, rating, upload date)
- Detects the base model (FLUX, SDXL, SD 1.5, Pony, Illustrious) and files the note in the matching subdirectory
- Skips images that have already been downloaded, so reruns are cheap
- Filters NSFW content if you want a SFW-only library
- Captures LoRA stacks from user-generated images, including filenames and weights
- Retries failed requests with backoff and applies configurable rate limits

## Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/civitai-to-obsidian.git
cd civitai-to-obsidian
```

Install dependencies. The project pins Python 3.13 via mise, so the recommended path is:

```bash
mise trust
mise install
mise exec -- pip install -r requirements.txt
```

If you'd rather use your system Python (3.9 or newer):

```bash
pip install -r requirements.txt
```

Copy the example config and edit it to point at your vault:

```bash
cp config.example.yaml config.yaml
$EDITOR config.yaml
```

At minimum, set `obsidian.vault_path` to your vault's root.

## Configuration

All settings live in `config.yaml`. Command-line flags override config values when both are set.

```yaml
obsidian:
  vault_path: "/path/to/your/vault"
  media_folder: "zzMedia/Model and Lora Example Images"
  base_directory: "Diffusion"

  # Notes are filed under base_directory/<base model>/<type>/
  directories:
    flux: "1 - FLUX"
    sd15: "2 - SD15"
    sdxl: "3 - SDXL"
    pony: "4 - PONY"
    illust: "5 - ILLUST"
    other: "9 - Other"

  type_directories:
    lora: "Lora"
    checkpoint: "Models"
    textualinversion: "Embeddings"
    other: "Other"

civitai:
  api_key: null  # paste your key here for higher rate limits
  base_url: "https://civitai.com/api/v1"

rate_limits:
  download_delay: 2.0
  api_delay: 1.5
  max_retries: 3
  backoff_factor: 1

defaults:
  image_limit: 200
  sort_order: "Most Reactions"
  time_period: "AllTime"
  nsfw_filter: "all"

metadata:
  author: "Your Name"
  base_tags:
    - "diffusion"
    - "ai"
    - "civitai"
```

The `base_tags` list is added to every generated note. The script also appends type-specific tags (`lora`, `diffusion`, base model, model name) automatically.

## Usage

Basic run with a model ID:

```bash
python civitai_to_obsidian.py 12345
```

Or with a CivitAI URL:

```bash
python civitai_to_obsidian.py https://civitai.com/models/12345/some-lora
```

### Targeting a specific version

Many models have multiple versions for different base models (SDXL, Flux, Pony, etc). Pass the full URL with the `modelVersionId` query parameter and the script will only fetch images for that version, and will detect the base model from that version when picking a directory and tags:

```bash
python civitai_to_obsidian.py "https://civitai.com/models/1155749?modelVersionId=1404932"
```

To find the version ID, click the version dropdown on the model page and copy the URL from the address bar.

### Sorting

```bash
python civitai_to_obsidian.py 12345 --sort "Most Reactions"   # default, official examples first
python civitai_to_obsidian.py 12345 --sort "Newest"           # latest user uploads
python civitai_to_obsidian.py 12345 --sort "Most Comments"
```

`Newest` is the better choice when you want to see how other people are stacking LoRAs, since user uploads tend to include more varied prompt and LoRA combinations than the creator's own examples.

### NSFW filter

```bash
python civitai_to_obsidian.py 12345 --nsfw all      # default, no filter
python civitai_to_obsidian.py 12345 --nsfw block    # SFW only
python civitai_to_obsidian.py 12345 --nsfw allow    # NSFW only
```

Tag-based filtering (anime, portrait, landscape, etc) isn't supported by the CivitAI API, so it isn't supported here either.

### Other useful flags

```bash
python civitai_to_obsidian.py 12345 --api-key YOUR_KEY        # use your CivitAI API key
python civitai_to_obsidian.py 12345 --limit 300               # fetch more images
python civitai_to_obsidian.py 12345 --vault-path /some/path   # override vault path
python civitai_to_obsidian.py 12345 --delay 3.0 --api-delay 2.0  # back off on rate limits
python civitai_to_obsidian.py 12345 --skip-download           # generate the page without downloading
python civitai_to_obsidian.py 12345 --config /some/config.yaml
```

Run `python civitai_to_obsidian.py --help` to see everything.

## Getting a CivitAI API key

1. Sign in at https://civitai.com/user/account
2. Scroll to the API Keys section
3. Click "Add API Key" and copy the value

Either paste it into `config.yaml` under `civitai.api_key` or pass it as `--api-key`. Authenticated requests get roughly 5x the unauthenticated rate limit (around 500 vs 100 requests per minute).

## Output structure

Notes are filed under `<vault>/<base_directory>/<base model dir>/<type dir>/`. With the default config, an SDXL LoRA called "My Awesome LoRA" lands at:

```
<vault>/
├── Diffusion/
│   └── 3 - SDXL/
│       └── Lora/
│           └── SDXL - My Awesome Lora.md
└── zzMedia/
    └── Model and Lora Example Images/
        └── My Awesome LoRA/
            ├── 4567890.jpeg
            ├── 4567891.png
            └── 4567892.webp
```

When you target a specific version with `?modelVersionId=...`, the version name is appended to the image folder (e.g. `My Awesome LoRA (v2)`) so you can keep multiple versions side by side without them colliding. The note filename itself stays the same, so a second run for a different version will overwrite the page.

Image files are named after the CivitAI image ID and the extension is detected from the file's magic bytes rather than the URL, since the CDN serves PNGs and WEBPs from URLs ending in `.jpeg`. Videos are filtered out at the API level and rejected again at download time.

## What the generated page looks like

A YAML frontmatter block with tags and metadata, then the description, then each image embedded with its parameters. A trimmed example:

````markdown
---
tags:
  - diffusion
  - ai
  - civitai
  - lora
  - sdxl
  - my-awesome-lora
author: Your Name
created: 2026-05-16
source: https://civitai.com/models/12345
type: LORA
civitai creator: amazing_artist
downloads: 15,234
rating: 4.8/5
upload date: 2025-11-02
civitai tags: character, anime, style
---

# SDXL - My Awesome Lora

## Description

...

---

## Example Images

#### Image 1

![[zzMedia/Model and Lora Example Images/My Awesome LoRA/4567890.jpeg]]

*45 reactions | 1024×1536*

**Positive Prompt:**
```
masterpiece, best quality, mychar, special_style
```

**Negative Prompt:**
```
bad anatomy, worst quality, low quality
```

**Parameters:**
- **Model:** myModel_v3
- **sampler:** DPM++ 2M Karras
- **steps:** 30
- **cfgScale:** 7
- **seed:** 1234567890
- **Size:** 512x768
````

LoRA usage tends to come through user-uploaded images either as `<lora:name:weight>` syntax inside the prompt itself, or as separate metadata keys depending on whether the image was generated with Automatic1111, ComfyUI, or another tool. The script doesn't try to normalize these, it just renders whatever CivitAI returns under `**Parameters:**`. ComfyUI workflow blobs are stripped out because they make the notes unreadable.

## Rate limiting and rough timings

| Images | With API key | Without API key |
|--------|--------------|-----------------|
| 10     | ~30s         | ~30s            |
| 50     | ~2 min       | ~3 min          |
| 100    | ~4 min       | ~6 min          |
| 200    | ~8 min       | ~12 min         |

Defaults (`download_delay: 2.0`, `api_delay: 1.5`) are conservative and aimed at unauthenticated use. With an API key you can usually drop them to `1.0` and `0.5` without trouble.

## Batch processing

A simple loop over a list of model IDs is enough for most cases. Sleep a few seconds between models to be polite:

```bash
#!/bin/bash
# batch_fetch.sh

models=(
    12345
    67890
    11223
)

for model in "${models[@]}"; do
    echo "Processing model: $model"
    python civitai_to_obsidian.py "$model" \
        --api-key "$CIVITAI_API_KEY" \
        --limit 50 \
        --sort "Newest" \
        --nsfw block

    sleep 5
done
```

Already-downloaded images are skipped on rerun, so it's safe to interrupt and resume.

## Troubleshooting

**"Could not extract model ID from: ..."**
The argument needs to be a numeric ID or a URL of the form `https://civitai.com/models/<id>` (optionally with `?modelVersionId=<vid>`).

**Rate limit errors (HTTP 429)**
Add an API key, or raise `--delay` and `--api-delay`. The script retries 429s with exponential backoff, but a sustained burst will still hit the wall.

**Images aren't showing up in Obsidian**
Check that `vault_path` is correct and that the media folder exists in your vault. Make sure "Use [[Wikilinks]]" is enabled under Settings → Files & Links. If links resolve to the wrong file, try toggling "New link format" between "Shortest path when possible" and "Absolute path in vault".

**Lots of images come back without metadata**
That's normal for user uploads. The script reports `Fetched X images (Y with generation metadata)` so you can see how many had usable params. Switch to `--sort "Most Reactions"` if you want creator examples (those almost always have metadata), or raise `--limit` to cast a wider net.

**The model has multiple base model versions and the wrong one is being detected**
Use the URL form with `?modelVersionId=<id>` to pin the version explicitly. Without it, the script uses the first version returned by the API, which is usually but not always the latest one.

## Repo layout

- `civitai_to_obsidian.py` is the main script
- `config.example.yaml` is the template config
- `requirements.txt` lists Python dependencies
- `.mise.toml` pins the Python version for mise users
- `fix_existing_images.py` and `rename_model_folders.py` are one-off maintenance helpers for cleaning up an existing library
- `inspect_image_meta.py` and `inspect_model_data.py` are small debugging scripts for poking at the API response shape
