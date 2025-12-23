import os
import sys
import torch
import torch.nn as nn
import numpy as np
import tifffile
import math
import zipfile
from tqdm import tqdm
from pathlib import Path

# ====================================================
# 1. MODEL ARCHITECTURE (AttentionUNet3D)
# ====================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        g1 = min(groups, out_channels)
        self.gn1 = nn.GroupNorm(g1, out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        g2 = min(groups, out_channels)
        self.gn2 = nn.GroupNorm(g2, out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        return self.act2(self.conv2(self.gn2(self.act1(self.gn1(self.conv1(x))))) + residual)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1), nn.GroupNorm(1, F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1), nn.GroupNorm(1, F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, kernel_size=1), nn.GroupNorm(1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16, attention=True, depth=3, pool_kernel_size=(2, 2, 2)):
        super().__init__()
        self.attention = attention
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        features = init_features
        curr_in = in_channels
        for i in range(depth):
            self.encoders.append(ResidualBlock(curr_in, features))
            self.pools.append(nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_kernel_size))
            curr_in = features
            features *= 2
        
        self.bottleneck = ResidualBlock(curr_in, features)
        
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        if self.attention: self.attentions = nn.ModuleList()
            
        for i in range(depth):
            self.up_convs.append(nn.ConvTranspose3d(features, features // 2, kernel_size=pool_kernel_size, stride=pool_kernel_size))
            if self.attention:
                self.attentions.append(AttentionBlock(F_g=features//2, F_l=features//2, F_int=features//4))
            self.decoders.append(ResidualBlock(features, features // 2))
            features //= 2
        self.final = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pools[i](x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.up_convs[i](x)
            skip = enc_features[self.depth - 1 - i]
            if self.attention: skip = self.attentions[i](g=x, x=skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
        return self.final(x)

# ====================================================
# 2. INFERENCE UTILS (Sliding Window)
# ====================================================
def predict_sliding_window(model, volume, patch_size=(32, 128, 128), overlap=0.25, batch_size=4, device='cuda'):
    model.eval()
    D, H, W = volume.shape
    d, h, w = patch_size
    stride_d, stride_h, stride_w = int(d*(1-overlap)), int(h*(1-overlap)), int(w*(1-overlap))
    
    steps_d = math.ceil((D - d) / stride_d) + 1 if D > d else 1
    steps_h = math.ceil((H - h) / stride_h) + 1 if H > h else 1
    steps_w = math.ceil((W - w) / stride_w) + 1 if W > w else 1
    
    prob_map = np.zeros((D, H, W), dtype=np.float16)
    count_map = np.zeros((D, H, W), dtype=np.uint8)
    
    patches, coords = [], []
    for z_step in range(steps_d):
        for y_step in range(steps_h):
            for x_step in range(steps_w):
                z, y, x = z_step*stride_d, y_step*stride_h, x_step*stride_w
                if z+d > D: z = D-d
                if y+h > H: y = H-h
                if x+w > W: x = W-w
                if z<0: z=0; 
                if y<0: y=0; 
                if x<0: x=0
                patches.append(volume[z:z+d, y:y+h, x:x+w])
                coords.append((z, y, x))
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_p = np.stack(patches[i:i+batch_size])
            batch_t = torch.from_numpy(batch_p).unsqueeze(1).to(device)
            with torch.cuda.amp.autocast():
                probs = torch.sigmoid(model(batch_t)).cpu().numpy()
            
            for j, (z, y, x) in enumerate(coords[i:i+batch_size]):
                prob_map[z:z+d, y:y+h, x:x+w] += probs[j, 0].astype(np.float16)
                count_map[z:z+d, y:y+h, x:x+w] += 1
                
    mask = count_map > 0
    prob_map[mask] /= count_map[mask]
    return prob_map.astype(np.float32)

# ====================================================
# 3. MAIN KAGGLE SUBMISSION LOOP
# ====================================================

# --- CONFIG ---
# UPDATE THIS PATH TO MATCH YOUR UPLOADED DATASET NAME
MODEL_WEIGHTS = '/kaggle/input/vesuvius-scroll-weights/best_topology_cldice.pth' 

# DEBUG: Check what datasets are actually attached
print("Available datasets in /kaggle/input/:")
if os.path.exists('/kaggle/input'):
    print(os.listdir('/kaggle/input'))
else:
    print("/kaggle/input does not exist! Are you running on Kaggle?")

# Competition Path - standard for Ink Detection
TEST_PATH = '/kaggle/input/vesuvius-challenge-ink-detection/test' 

# If the standard path doesn't exist, try to find it dynamically
if not os.path.exists(TEST_PATH):
    print(f"WARNING: {TEST_PATH} not found.")
    # Try generic search
    possible_paths = [
        '/kaggle/input/vesuvius-challenge-ink-detection/test',
        '/kaggle/input/vesuvius-challenge-2024/test', # If there's a new one
    ]
    for p in possible_paths:
        if os.path.exists(p):
            TEST_PATH = p
            print(f"Found test data at: {TEST_PATH}")
            break

OUTPUT_DIR = 'submission_tifs'
THRESHOLD = 0.3

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Model
model = AttentionUNet3D(attention=True, depth=3, pool_kernel_size=(1, 2, 2)).to(device)
if os.path.exists(MODEL_WEIGHTS):
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    print(f"Loaded weights from {MODEL_WEIGHTS}")
else:
    print(f"ERROR: Weights not found at {MODEL_WEIGHTS}. Please check dataset path.")

model.eval()

# Process Test Volumes
# Kaggle structure: /test/[fragment_id]/surface_volume/[slice.tif]...
test_fragments = sorted([d.name for d in Path(TEST_PATH).iterdir() if d.is_dir()])
print(f"Found fragments: {test_fragments}")

for frag_id in tqdm(test_fragments):
    frag_path = Path(TEST_PATH) / frag_id
    
    # Need to load volume. This handles individual slice TIFFs loading into 3D volume
    # Or single large TIFF? NOTE: Competition data is usually stack of .tif images
    # We need to stack them.
    surface_vol_path = frag_path / 'surface_volume'
    layers = sorted(list(surface_vol_path.glob('*.tif')))
    
    if not layers:
        print(f"No tifs found for {frag_id}")
        continue
        
    print(f"Loading {len(layers)} slices for {frag_id}...")
    # Read first to get shape
    sample = tifffile.imread(layers[0])
    h, w = sample.shape
    d = len(layers)
    
    # Pre-allocate volume (be careful with RAM on Kaggle!)
    volume = np.zeros((d, h, w), dtype=np.uint8)
    for i, layer_path in enumerate(layers):
        volume[i] = tifffile.imread(layer_path)
        
    # Normalize 8-bit
    volume = volume.astype(np.float32) / 255.0
    
    # Predict
    print(f"Predicting {frag_id}...")
    probs = predict_sliding_window(model, volume, patch_size=(32, 128, 128), overlap=0.25, device=device)
    
    # Threshold
    mask = (probs > THRESHOLD).astype(np.uint8)
    
    # Save
    tifffile.imwrite(f"{OUTPUT_DIR}/{frag_id}.tif", mask, compression='zlib')

# Create Zip
print("Zipping submission...")
with zipfile.ZipFile('submission.zip', 'w') as zf:
    for tif_file in Path(OUTPUT_DIR).glob('*.tif'):
        zf.write(tif_file, arcname=tif_file.name)

print("Done! submission.zip is ready.")
