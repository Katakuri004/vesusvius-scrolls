
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import tifffile
import cv2
import sys
sys.path.append('src')

from data.dataset import MultiVolumePatchDataset

DATA_DIR = Path('data')
TRAIN_IMG_DIR = DATA_DIR / 'train_images'
TRAIN_LBL_DIR = DATA_DIR / 'train_labels'

all_volumes = sorted(list(TRAIN_IMG_DIR.glob('*.tif')))
all_labels = [TRAIN_LBL_DIR / v.name for v in all_volumes]

print(f"Found {len(all_volumes)} volumes.")

# Init Dataset
ds = MultiVolumePatchDataset(all_volumes, all_labels, patch_size=(32, 128, 128), samples_per_epoch=10, pos_fraction=0.1)
loader = DataLoader(ds, batch_size=1)

print("\n--- Checking Batches ---")
debug_dir = Path('debug_data_check')
debug_dir.mkdir(exist_ok=True)

for i, batch in enumerate(loader):
    if i >= 5: break
    
    vol = batch['volume'].numpy()
    lbl = batch['label'].numpy()
    
    print(f"Batch {i}:")
    print(f"  Vol: Shape={vol.shape}, Min={vol.min():.4f}, Max={vol.max():.4f}, Mean={vol.mean():.4f}")
    print(f"  Lbl: Shape={lbl.shape}, Min={lbl.min():.4f}, Max={lbl.max():.4f}, Mean={lbl.mean():.4f}")
    
    # Save slices
    mid_z = vol.shape[2] // 2
    
    # Volume Slice
    v_slice = vol[0, 0, mid_z, :, :]
    
    # Label Slices (Check what is what)
    l_slice_raw = lbl[0, 0, mid_z, :, :]
    mask_0 = (l_slice_raw == 0).astype(np.uint8) * 255
    mask_1 = (l_slice_raw == 1).astype(np.uint8) * 255
    mask_2 = (l_slice_raw == 2).astype(np.uint8) * 255
    
    cv2.imwrite(str(debug_dir / f'batch_{i}_vol.png'), (v_slice * 255).astype(np.uint8))
    cv2.imwrite(str(debug_dir / f'batch_{i}_lbl_raw.png'), (l_slice_raw * 50).astype(np.uint8)) # Scale for vis
    cv2.imwrite(str(debug_dir / f'batch_{i}_mask_0.png'), mask_0)
    cv2.imwrite(str(debug_dir / f'batch_{i}_mask_1.png'), mask_1)
    cv2.imwrite(str(debug_dir / f'batch_{i}_mask_2.png'), mask_2)
    
print(f"\nSaved debug images to {debug_dir}")
