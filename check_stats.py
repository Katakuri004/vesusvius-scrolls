
import tifffile
import numpy as np
from pathlib import Path

data_dir = Path('data/train_images')
# Find first tif
tif_files = list(data_dir.glob('*.tif'))
if not tif_files:
    print("No TIF files found in data/train_images")
    exit()

path = tif_files[0]
print(f"Reading {path}...")
vol = tifffile.imread(path)
print(f"Shape: {vol.shape}")
print(f"Dtype: {vol.dtype}")
print(f"Min: {vol.min()}")
print(f"Max: {vol.max()}")
print(f"Mean: {vol.mean()}")
print(f"P01: {np.percentile(vol, 1)}")
print(f"P99: {np.percentile(vol, 99)}")
