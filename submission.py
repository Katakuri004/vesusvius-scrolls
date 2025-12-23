import torch
import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('src')

from models.unet3d import AttentionUNet3D
from utils.inference import predict_sliding_window

def rle_encode(mask):
    """
    RLE encoding for Vesuvius Challenge.
    Returns string: "start length start length ..."
    Note: Mask should be flattened.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def generate_submission(model_path, data_dir, output_zip='submission.zip', threshold=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = AttentionUNet3D(attention=True, depth=3, pool_kernel_size=(1, 2, 2)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Identify volumes
    test_dir = Path(data_dir) / 'test'
    if test_dir.exists():
        volume_paths = sorted(list(test_dir.glob('*.tif')))
    else:
        print("Test directory not found. Using Validation volumes for demo.")
        train_dir = Path(data_dir) / 'train_images'
        all_vols = sorted(list(train_dir.glob('*.tif')))
        volume_paths = [all_vols[-1]] # Demo on last volume

    import zipfile
    output_dir = Path("submission_tifs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Generating TIFF submission to {output_zip}...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for v_path in tqdm(volume_paths, desc="Processing Volumes"):
            vol_id = v_path.stem
            
            try:
                # Read Volume
                vol = tifffile.imread(v_path).astype(np.float32)
                # Normalize (Auto-detect)
                if vol.max() <= 255:
                    vol /= 255.0
                else:
                    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
                    
                # Predict
                with torch.no_grad():
                    probs = predict_sliding_window(model, vol, patch_size=(32, 128, 128), overlap=0.25, device=device)
                
                # Threshold & Type Cast
                # Requirement: "match the dimensions ... use the same data type as the train mask"
                # Train mask is uint8. 
                mask = (probs > threshold).astype(np.uint8)
                
                # Save TIFF
                tif_name = f"{vol_id}.tif"
                tif_path = output_dir / tif_name
                
                # Use standard compression to keep size down, or no compression if strictly required?
                # Usually .tif masks are compressed.
                tifffile.imwrite(tif_path, mask, compression='zlib')
                
                # Add to Zip
                zf.write(tif_path, arcname=tif_name)
                print(f"  Processed {vol_id}")
                
            except Exception as e:
                print(f"Error processing {vol_id}: {e}")

    print(f"Submission saved to {output_zip}")

if __name__ == "__main__":
    # Point to the best checkpoint
    ckpt = 'models/best_topology_cldice.pth' 
    if not Path(ckpt).exists():
        ckpt = 'models/best_dice_cldice.pth' # Fallback
        
    generate_submission(ckpt, 'data')
