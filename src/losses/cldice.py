import torch
import torch.nn as nn
import torch.nn.functional as F

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        y_pred: (B, C, D, H, W) Logits or Probs
        y_true: (B, C, D, H, W) Binary Mask
        """
        # Ensure probs
        if not ((0 <= y_pred).all() and (y_pred <= 1).all()):
             skel_pred = torch.sigmoid(y_pred)
        else:
             skel_pred = y_pred

        skel_true = soft_skeletonize(y_true, iter_=self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, skel_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        
        skel_pred = soft_skeletonize(skel_pred, iter_=self.iter)
        tsens = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(y_true[:,1:,...])+self.smooth)    

        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img, (3,1,1), (1,1,1), (1,0,0))
        p2 = -F.max_pool3d(-img, (1,3,1), (1,1,1), (0,1,0))
        p3 = -F.max_pool3d(-img, (1,1,3), (1,1,1), (0,0,1))
        return torch.min(torch.min(p1, p2), p3)

def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img-img1)
    for i in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img-img1)
        skel = skel + F.relu(delta-skel*delta)
    return skel

def soft_skeletonize(x, iter_=3):
    return soft_skel(x, iter_)
