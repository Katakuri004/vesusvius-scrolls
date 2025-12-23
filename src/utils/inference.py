import torch
import numpy as np
from tqdm import tqdm
import math

def predict_sliding_window(model, volume, patch_size=(32, 128, 128), overlap=0.5, batch_size=4, device='cuda'):
    """
    Performs sliding window inference on a 3D volume.
    
    Args:
        model: PyTorch model (expects 5D input: B,C,D,H,W)
        volume: 3D numpy array (D, H, W) normalized [0,1]
        patch_size: Tuple (d, h, w)
        overlap: Overlap fraction (0.0 to 1.0). 0.5 recommended.
        batch_size: Inference batch size
        device: 'cuda' or 'cpu'
        
    Returns:
        prob_map: 3D numpy array (D, H, W) with predicted probabilities [0,1]
    """
    model.eval()
    D, H, W = volume.shape
    d, h, w = patch_size
    
    # Calculate strides
    stride_d = int(d * (1 - overlap))
    stride_h = int(h * (1 - overlap))
    stride_w = int(w * (1 - overlap))
    
    # Calculate number of steps
    steps_d = math.ceil((D - d) / stride_d) + 1 if D > d else 1
    steps_h = math.ceil((H - h) / stride_h) + 1 if H > h else 1
    steps_w = math.ceil((W - w) / stride_w) + 1 if W > w else 1
    
    # Output buffers
    # We use a count buffer to average overlapping predictions
    prob_map = np.zeros((D, H, W), dtype=np.float16) # FP16 to save RAM
    count_map = np.zeros((D, H, W), dtype=np.uint8)
    
    patches = []
    coords = []
    
    # Generate Patches
    for z_step in range(steps_d):
        for y_step in range(steps_h):
            for x_step in range(steps_w):
                z = z_step * stride_d
                y = y_step * stride_h
                x = x_step * stride_w
                
                # Boundary Checks (Shift back if going over)
                if z + d > D: z = D - d
                if y + h > H: y = H - h
                if x + w > W: x = W - w
                
                # Safety for Volume < Patch
                if z < 0: z = 0
                if y < 0: y = 0
                if x < 0: x = 0
                
                patches.append(volume[z:z+d, y:y+h, x:x+w])
                coords.append((z, y, x))
                
    # Run Inference in Batches
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            batch_coords = coords[i:i+batch_size]
            
            # Stack and conform to (B, 1, D, H, W)
            # Handle potential edge case where volume < patch (padding) - omitted for now as volumes assumed large
            batch_tensor = np.stack(batch_patches)
            batch_tensor = torch.from_numpy(batch_tensor).unsqueeze(1).to(device) # Add Channel
            
            with torch.amp.autocast('cuda'):
                logits = model(batch_tensor)
                probs = torch.sigmoid(logits)
                
            probs_np = probs.cpu().numpy()
            
            # Place back
            for j, (z, y, x) in enumerate(batch_coords):
                prob_patch = probs_np[j, 0] # (D, H, W)
                
                # Accumulate
                # Note: This is simple averaging. Gaussian weighting matches center better.
                # For validation speed, simple average is okay.
                prob_map[z:z+d, y:y+h, x:x+w] += prob_patch.astype(np.float16)
                count_map[z:z+d, y:y+h, x:x+w] += 1
                
    # Average
    # Avoid division by zero
    mask = count_map > 0
    prob_map[mask] /= count_map[mask]
    
    return prob_map.astype(np.float32)
