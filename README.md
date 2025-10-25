# CivitAI to Obsidian Image Library Builder

Problem: When you pull down a new Model or Lora from CivitAI that you with to use it is helpful to also pull down example images and their generation paramaters to reference so you 
can see what the model / Lora is capable of and what styles it is able to produce. Obsidian is a very good option for this type of documentation but the documentation process can be 
incredably time consuming if done manually. 

Solution: This script automatically fetches example images from CivitAI models/LoRAs along with all their generation parameters and creates a comprehensive Obsidian markdown 
documentation page for you.

<img width="2157" height="1981" alt="image" src="https://github.com/user-attachments/assets/a12cd072-7d45-48b5-83db-be7f2ce2ccde" />


## Features

- ✅ Fetches images from model versions (avoids API timeout issues)
- ✅ Downloads images to your Obsidian vault's attachments folder
- ✅ Extracts all generation parameters (prompts, sampler, CFG scale, steps, **LoRAs**, etc.)
- ✅ Creates beautifully formatted Obsidian markdown pages
- ✅ Automatic retry logic and configurable rate limiting
- ✅ Skip already downloaded images
- ✅ Organizes images in folders by model name
- ✅ **Choose between official examples or user-generated images**
- ✅ **Detects and captures LoRA usage with weights**
- ✅ **NSFW content filtering (block, allow, or all)**

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/civitai-to-obsidian.git
cd civitai-to-obsidian
```

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

Or with mise (recommended for isolated environment):

```bash
mise trust
mise install
mise exec -- pip install -r requirements.txt
```

1. **Configure the script:**

```bash
# Copy the example config to create your personal config
cp config.example.yaml config.yaml

# Edit config.yaml with your settings
nano config.yaml  # or use your preferred editor
```

At minimum, update the `vault_path` in `config.yaml`:

```yaml
obsidian:
  vault_path: "/path/to/your/obsidian/vault"
```

## Configuration

The script uses a `config.yaml` file for all settings. This keeps your personal paths and preferences separate from the code.

### Configuration File Structure

```yaml
# Obsidian vault settings
obsidian:
  vault_path: "/path/to/your/vault"
  media_folder: "zzMedia/Model and Lora Example Images"
  base_directory: "Diffusion"

# CivitAI API settings
civitai:
  api_key: null  # Add your API key here

# Rate limiting
rate_limits:
  download_delay: 2.0
  api_delay: 1.5

# Default options
defaults:
  image_limit: 200
  sort_order: "Most Reactions"
  nsfw_filter: "all"

# Metadata
metadata:
  author: "Your Name"
  base_tags:
    - "ai"
    - "civitai"
```

### Configuration Options

- **vault_path**: Path to your Obsidian vault (required)
- **media_folder**: Where images are stored (relative to vault)
- **base_directory**: Where notes are organized (relative to vault)
- **api_key**: Your CivitAI API key for higher rate limits
- **download_delay**: Seconds between image downloads
- **api_delay**: Seconds between API calls
- **image_limit**: Default number of images to fetch
- **author**: Your name for note metadata
- **base_tags**: Tags added to all generated notes

Command-line arguments override config file settings.

## Usage

### Basic Usage (Official Creator Examples)

```bash
python civitai_to_obsidian.py 12345
```

### User-Generated Images (More LoRA Examples)

```bash
python civitai_to_obsidian.py 12345 --sort "Newest"
```

### Filter NSFW Content

```bash
# Block NSFW content (SFW only)
python civitai_to_obsidian.py 12345 --nsfw block

# Allow NSFW content only
python civitai_to_obsidian.py 12345 --nsfw allow

# No filtering (default)
python civitai_to_obsidian.py 12345 --nsfw all
```

### With CivitAI URL

```bash
python civitai_to_obsidian.py https://civitai.com/models/12345/my-awesome-lora
```

### Specific Model Version (for multi-version models)

```bash
# For models with multiple versions (SDXL, Flux, Pony, etc.)
# Use the modelVersionId parameter to target a specific version
python civitai_to_obsidian.py "https://civitai.com/models/1155749?modelVersionId=1404932"

# This is useful when a model has multiple versions for different base models
# The script will only fetch images for the specified version
```

### With API Key (for higher rate limits)

```bash
python civitai_to_obsidian.py 12345 --api-key YOUR_CIVITAI_API_KEY
```

### Fetch more images

```bash
python civitai_to_obsidian.py 12345 --limit 300
```

### Custom vault path

```bash
python civitai_to_obsidian.py 12345 --vault-path "/path/to/vault"
```

### Adjust rate limiting for heavy usage

```bash
python civitai_to_obsidian.py 12345 --delay 3.0 --api-delay 2.0
```

### Test without downloading images

```bash
python civitai_to_obsidian.py 12345 --skip-download
```

## Getting a CivitAI API Key (Optional but Recommended)

1. Go to <https://civitai.com/user/account>
2. Scroll down to "API Keys"
3. Click "Add API Key"
4. Copy your key and use it with `--api-key`

Having an API key gives you higher rate limits (500 req/min vs 100 req/min).

## Sorting Options

- **`--sort "Most Reactions"`** (default) - Official example images from model creator
- **`--sort "Newest"`** - Latest user-generated images (best for finding LoRA usage!)
- **`--sort "Most Comments"`** - Most discussed images

## Content Filtering

Use the `--nsfw` parameter to control content:

- **`--nsfw all`** (default) - No filtering, gets all images
- **`--nsfw block`** - SFW only, blocks NSFW content
- **`--nsfw allow`** - NSFW content only

**Note:** Tag-based filtering (e.g., filtering by "anime", "portrait", "landscape") is not supported by the CivitAI API.

### Example Usage

```bash
# Get only SFW images for a model
python civitai_to_obsidian.py 12345 --nsfw block --limit 100

# Build a library with only SFW content
python civitai_to_obsidian.py 12345 --nsfw block --sort "Most Reactions"
```

## Rate Limiting & Best Practices

### Without API Key

- ~100 requests/minute
- Default settings: `--delay 2.0 --api-delay 1.5` = ~20-30 images/min
- **Safe for:** 50-100 images per model

### With API Key

- ~500 requests/minute
- You can reduce delays: `--delay 1.0 --api-delay 0.5` = ~50-60 images/min
- **Safe for:** 200-400 images per model

### Building Large Libraries

For processing many models, create a batch script with delays between models:

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
        --api-key YOUR_KEY \
        --limit 50 \
        --sort "Newest" \
        --delay 2.0

    # 5 second delay between models
    echo "Waiting before next model..."
    sleep 5
done
```

## How It Works

1. **Fetches model details** - Gets model name, creator, stats, tags, trigger words
2. **Fetches images by version** - Uses `modelVersionId` to avoid timeouts
   - If a specific version ID is provided in the URL, only fetches images for that version
   - Otherwise, fetches images across all versions until limit is reached
3. **Downloads images** - Grabs images with progress tracking
4. **Extracts metadata** - Pulls generation params including **LoRAs with weights**
5. **Creates Obsidian page** - Generates a markdown file with:
   - Model information header
   - Description
   - All images embedded with their full generation parameters
   - **LoRA names and weights clearly listed**

## Output Structure

```text
/path/to/vault/
├── CivitAI - My_Awesome_LoRA.md          # Main documentation page
└── zzMedia/
    └── Model and Lora Example Images/
        └── My_Awesome_LoRA/              # Images folder
            ├── My_Awesome_LoRA_123456.jpeg
            ├── My_Awesome_LoRA_123457.jpeg
            └── ...
```

## Example Output

The generated Obsidian page includes:

```markdown
# My Awesome LoRA

## Model Information

- **Type:** LORA
- **Creator:** amazing_artist
- **Downloads:** 15,234
- **Rating:** 4.8/5
- **CivitAI Link:** https://civitai.com/models/12345
- **Tags:** character, anime, style
- **Trigger Words:** `mychar, special_style`

## Example Images (50 images)

### Image 1

![[zzMedia/Model and Lora Example Images/My_Awesome_LoRA/My_Awesome_LoRA_123456.jpeg]]

*45 reactions | 1024×1536*

**Positive Prompt:**
```text
masterpiece, best quality, mychar, special_style
```

**Negative Prompt:**

```text
bad anatomy, worst quality, low quality

**Parameters:**

- **Model:** myModel_v3
- **sampler:** DPM++ 2M Karras
- **steps:** 30
- **cfgScale:** 7
- **seed:** 1234567890
- **Size:** 512x768
- **Hires upscale:** 2
- **LoRAs Used:**
  - EMS-533410-EMS.safetensors (weight: 1.0)
  - detail-tweaker.safetensors (weight: 0.8)

```

## LoRA Detection

The script automatically captures LoRA information from user-generated images! Look for entries like:

```text
"text": "<lora:EMS-533410-EMS.safetensors:1.000000>,<lora:EMS-654444-FP8-EMS.safetensors:1.100000>"
```

This shows:

- LoRA filenames
- Weights used (1.0, 1.1, etc.)
- Multiple LoRAs stacked together

**Tip:** Use `--sort "Newest"` to get more user-generated images, which often include additional LoRAs!

## Working with Multi-Version Models

Many models on CivitAI have multiple versions for different base models (SDXL, Flux, Pony, SD 1.5, Illustrious, etc.). You can target a specific version using the `modelVersionId` parameter in the URL.

### Finding the Model Version ID

1. Visit the model page on CivitAI
2. Click on the version dropdown (e.g., "SDXLv2", "FluxV2", "PonyV2")
3. Copy the URL which will contain `?modelVersionId=XXXXXX`
4. Use that full URL with the script

### Example

```bash
# Model with SDXL, Flux, Pony, SD15, and Illustrious versions
# Target only the SDXL version:
python civitai_to_obsidian.py "https://civitai.com/models/1155749?modelVersionId=1404932"

# Without modelVersionId, it would fetch images from all versions
# With modelVersionId, it only fetches from the SDXLv2 version
```

### Why This Matters

- **Accurate documentation**: Each version may have different capabilities
- **Correct base model detection**: Ensures files are organized in the right directory
- **Focused examples**: Get only the images relevant to the version you're using
- **Efficiency**: Don't waste time downloading images from versions you won't use

### Output Organization

When you specify a version ID, the script:

1. Detects the base model from that specific version
2. Saves the note in the correct directory (e.g., `3 - SDXL/Lora/`)
3. Prefixes the title with the base model (e.g., "SDXL - Model Name")
4. Tags it with the appropriate base model tag

## Troubleshooting

### "Could not extract model ID"

- Make sure you're using a valid CivitAI URL or numeric model ID

### "Rate limit exceeded"

- Add `--api-key YOUR_KEY` to get higher limits
- Increase delays: `--delay 3.0 --api-delay 2.0`
- Reduce `--limit` to fetch fewer images per model

### "Images not showing in Obsidian"

- Check that the vault path is correct
- Make sure the `zzMedia/Model and Lora Example Images/` folder exists
- Verify "Use [[Wikilinks]]" is enabled in Obsidian settings
- Try toggling between "Shortest path" and "Absolute path" in Settings → Files & Links

### "Fetched images but many don't have metadata"

- This is normal for user-uploaded images
- The script reports: "Fetched X images (Y with generation metadata)"
- Use `--sort "Most Reactions"` for official examples (always have metadata)
- Or increase `--limit` to get more images and find ones with metadata

## Tips for Building a Large Library

1. **Start with moderate limits:** Test with `--limit 20` first
2. **Use API key:** Get 5x higher rate limits
3. **Mix sorting options:**
   - Official examples: `--sort "Most Reactions"`
   - User LoRA combos: `--sort "Newest"`
4. **Be patient:** 2-second delays mean ~30 images/minute
5. **Batch process:** Use a script to process multiple models with delays between them
6. **Resume capability:** The script skips already-downloaded images automatically
7. **Content filtering:** Use `--nsfw block` to build SFW-only libraries

## Advanced: Batch Processing

Process multiple models efficiently:

```bash
#!/bin/bash
# process_favorites.sh

# Read model IDs from file
while IFS= read -r model_id; do
    echo "===================================="
    echo "Processing: $model_id"
    echo "===================================="

    python civitai_to_obsidian.py "$model_id" \
        --api-key "$CIVITAI_API_KEY" \
        --limit 50 \
        --sort "Newest" \
        --nsfw block \
        --delay 2.0

    echo "Waiting 10 seconds before next model..."
    sleep 10
done < model_ids.txt

echo "All models processed!"
```

Create `model_ids.txt`:

```text
4201
12345
67890
```

Then run: `bash process_favorites.sh`

## Performance Estimates

| Images | With API Key | Without API Key |
|--------|-------------|----------------|
| 10     | ~30 seconds | ~30 seconds    |
| 50     | ~2 minutes  | ~3 minutes     |
| 100    | ~4 minutes  | ~6 minutes     |
| 200    | ~8 minutes  | ~12 minutes    |

*Times include API calls + image downloads

## Support

If you run into issues, check:

1. Python version (3.7+)
2. Dependencies installed (`pip install -r requirements.txt`)
3. Valid CivitAI model URL or ID
4. Correct vault path
5. API key is valid (if using one)

Enjoy building your CivitAI image library! 🎨
