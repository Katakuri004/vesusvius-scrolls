import torch
import numpy as np
from cc3d import connected_components

def get_connected_components(mask, connectivity=26):
    """
    Computes connected components for a binary 3D mask.
    Returns: labeled_mask, num_components
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(np.uint8)
    labels, N = connected_components(mask, connectivity=connectivity, return_N=True)
    return labels, N

def calculate_merge_score(pred_prob, target, threshold=0.5, min_overlap=20):
    """
    Computes the Merge Score: Number of predicted components that bridge 
    (overlap with) at least 2 distinct Ground Truth components.
    
    High Merge Score = Bad Topology (Layers are merging).
    """
    pred_mask = (pred_prob > threshold).astype(np.uint8)
    target_mask = (target > 0).astype(np.uint8)

    # Label components
    pred_labels, num_pred = get_connected_components(pred_mask)
    target_labels, num_target = get_connected_components(target_mask)
    
    if num_pred == 0 or num_target == 0:
        return 0

    merge_count = 0
    
    # Iterate over each predicted component
    for i in range(1, num_pred + 1):
        # Extract binary mask for this P_comp
        p_loc = (pred_labels == i)
        
        # Find which GT labels it overlaps with
        overlapping_gt_labels = np.unique(target_labels[p_loc])
        
        # Remove 0 (background)
        overlapping_gt_labels = overlapping_gt_labels[overlapping_gt_labels != 0]
        
        # If it overlaps >= 2 distinct GT components, it's a merge!
        if len(overlapping_gt_labels) >= 2:
            # Check overlap size (optional robustness)
            valid_overlaps = 0
            for gt_id in overlapping_gt_labels:
                intersection = np.sum((target_labels == gt_id) & p_loc)
                if intersection > min_overlap:
                    valid_overlaps += 1
            
            if valid_overlaps >= 2:
                merge_count += 1
                
    return merge_count

def calculate_shatter_score(pred_prob, threshold=0.5):
    """
    Computes Shatter Score: Size of Largest Component / Total Foreground Size.
    
    Low Score (< 0.5) = Shattered predictions (Dust/Noise).
    High Score (> 0.8) = Coherent predictions.
    """
    pred_mask = (pred_prob > threshold).astype(np.uint8)
    total_fg = np.sum(pred_mask)
    
    if total_fg == 0:
        return 0.0
    
    labels, N = get_connected_components(pred_mask)
    
    if N == 0:
        return 0.0
        
    counts = np.bincount(labels.flat)
    # counts[0] is background, ignore it.
    if len(counts) > 1:
        largest_cc_size = counts[1:].max()
    else:
        return 0.0
        
    return largest_cc_size / total_fg

def calculate_dice(pred_prob, target, threshold=0.5, smooth=1e-6):
    pred_mask = (pred_prob > threshold).float()
    target_mask = (target > 0).float()
    
    intersection = (pred_mask * target_mask).sum()
    return (2. * intersection + smooth) / (pred_mask.sum() + target_mask.sum() + smooth)
