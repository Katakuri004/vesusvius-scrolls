import numpy as np
from cc3d import connected_components
import scipy.ndimage as ndimage

def get_connected_components(mask, connectivity=26):
    mask = mask.astype(np.uint8)
    labels, N = connected_components(mask, connectivity=connectivity, return_N=True)
    return labels, N

def filter_small_components(mask, min_size=500):
    """
    Removes connected components smaller than min_size voxels.
    """
    labels, N = get_connected_components(mask)
    if N == 0:
        return mask
    
    # Bincount is fast for counting
    counts = np.bincount(labels.flat)
    
    # Create a mask of labels to keep
    # counts[0] is background. We want counts > min_size
    keep_labels = np.where(counts > min_size)[0]
    keep_labels = keep_labels[keep_labels != 0] # Remove background
    
    # Create cleaner mask
    clean_mask = np.isin(labels, keep_labels).astype(np.uint8)
    return clean_mask

def break_bridges(mask, iterations=1):
    """
    Applies Morphological Opening (Erosion -> Dilation) to break thin connections.
    """
    # Structuring element: 3x3x3 ball-ish
    struct = ndimage.generate_binary_structure(3, 1) 
    
    # Erode to snap bridges
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=iterations)
    
    # Dilate back to restore volume key shape (but bridges remain broken)
    # We dilate slightly less or same to avoid re-bridging? 
    # Usually standard opening is equal iterations.
    opened = ndimage.binary_dilation(eroded, structure=struct, iterations=iterations)
    
    return opened.astype(np.uint8)

def clean_prediction(prob_vol, threshold=0.3, min_component_size=1000, bridge_break_iter=1):
    """
    Full pipeline: Threshold -> Break Bridges -> Filter Dust
    """
    # 1. Threshold
    mask = (prob_vol > threshold).astype(np.uint8)
    
    # 2. Break Bridges (Morph Opening)
    if bridge_break_iter > 0:
        mask = break_bridges(mask, iterations=bridge_break_iter)
        
    # 3. Filter Dust
    mask = filter_small_components(mask, min_size=min_component_size)
    
    return mask
