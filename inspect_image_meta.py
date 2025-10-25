#!/usr/bin/env python3
"""Check if image metadata is in model response or needs separate fetch"""

import requests
import json

model_id = 4201
response = requests.get(f"https://civitai.com/api/v1/models/{model_id}", timeout=10)
data = response.json()

version = data['modelVersions'][0]
img = version['images'][0]

print("Complete first image data:")
print(json.dumps(img, indent=2))

print("\n" + "="*60)
print("\nChecking if 'meta' field exists:")
if 'meta' in img:
    print("✓ Yes! Metadata is included")
    print("\nMetadata keys:", list(img['meta'].keys()) if img.get('meta') else None)
    if img.get('meta'):
        print(json.dumps(img['meta'], indent=2)[:800])
else:
    print("✗ No 'meta' field in image data")
    print("  We'll need to fetch each image individually from /api/v1/images/{id}")
