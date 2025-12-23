import torch
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('src')

from models.unet3d import AttentionUNet3D
from utils.inference import predict_sliding_window

def generate_showcase():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    img_dir = Path("img")
    img_dir.mkdir(exist_ok=True)
    
    ckpt = 'models/best_topology_cldice.pth'
    if not Path(ckpt).exists():
        ckpt = 'models/best_dice_cldice.pth'
    
    # Load Model
    model = AttentionUNet3D(attention=True, depth=3, pool_kernel_size=(1, 2, 2)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    
    # Load Data (Validation Volume)
    data_dir = Path("data/train_images")
    lbl_dir = Path("data/train_labels")
    
    # Use the last volume as it was likely our validation set
    v_path = sorted(list(data_dir.glob('*.tif')))[-1]
    l_path = lbl_dir / v_path.name
    
    print(f"Generating showcase from {v_path.name}")
    
    # Read Volume
    vol = tifffile.imread(v_path).astype(np.float32)
    if vol.max() <= 255:
        vol /= 255.0
    else:
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        
    gt = tifffile.imread(l_path)
    gt = (gt == 1).astype(np.uint8) # Ink only
    
    # Predict (Full Volume or Chunk)
    # To be fast, let's just pick a chunk around the middle
    mid_z = vol.shape[0] // 2
    z_start = max(0, mid_z - 16)
    z_end = min(vol.shape[0], mid_z + 16)
    
    # We need to run inference on a chunk that fits the model logic
    # The sliding window handles full volume, so let's just run it on the slice of interest context
    # But for 3D context, we pass the whole volume (or at least a good chunk)
    # Let's run on full volume to be safe and accurate, or crop z?
    # Cropping Z helps speed.
    
    chunk_vol = vol[z_start:z_end]
    
    with torch.no_grad():
        preds = predict_sliding_window(model, chunk_vol, patch_size=(32, 128, 128), overlap=0.25, device=device)
        
    # Extract middle slice of the prediction
    # mid_z in chunk is roughly 16
    chunk_mid = preds.shape[0] // 2
    
    pred_slice = preds[chunk_mid]
    gt_slice = gt[z_start + chunk_mid]
    in_slice = vol[z_start + chunk_mid]
    
    # Threshold
    pred_bin = (pred_slice > 0.3).astype(np.uint8)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(in_slice, cmap='gray')
    axes[0].set_title('Input CT Slice')
    axes[0].axis('off')
    
    # Overlay Pred on GT? Or just side by side?
    # Let's do Side by Side
    axes[1].imshow(gt_slice, cmap='gray')
    axes[1].set_title('Ground Truth (Ink)')
    axes[1].axis('off')
    
    axes[2].imshow(pred_bin, cmap='gray')
    axes[2].set_title('Prediction (Dice ~0.44)')
    axes[2].axis('off')
    
    save_path = img_dir / "model_output_showcase.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved showcase image to {save_path}")

if __name__ == "__main__":
    generate_showcase()
