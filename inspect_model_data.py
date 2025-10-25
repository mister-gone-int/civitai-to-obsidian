#!/usr/bin/env python3
"""Inspect what image data is available in model details"""

import requests
import json

model_id = 4201
response = requests.get(f"https://civitai.com/api/v1/models/{model_id}", timeout=10)
data = response.json()

print(f"Model: {data.get('name')}\n")
print(f"Model has {len(data.get('modelVersions', []))} versions\n")

if data.get('modelVersions'):
    version = data['modelVersions'][0]
    print(f"First version: {version.get('name')}")
    print(f"Version has {len(version.get('images', []))} images\n")

    if version.get('images'):
        img = version['images'][0]
        print("First image structure:")
        print(json.dumps(img, indent=2)[:500])
        print("\n...")
        print(f"\nTotal images across all versions: {sum(len(v.get('images', [])) for v in data['modelVersions'])}")
