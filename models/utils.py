import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import collections

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if classname.find('KernelConv2D') == -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.04)
        torch.nn.init.constant_(m.bias.data, 0)

def norm_feat_vis(feat):
    l2norm = torch.sqrt(np.finfo(float).eps + torch.sum(torch.mul(feat,feat),1,keepdim=True))
    feat = feat / (l2norm + np.finfo(float).eps)
    feat = feat.permute(1, 0, 2, 3)
    return feat

def norm_res_vis(res):
    res = res.detach().clone()
    b, c, h, w = res.size()

    res = res.view(res.size(0), -1)
    res = res - res.min(1, keepdim=True)[0]
    res = res / res.max(1, keepdim=True)[0]
    res = res.view(b, c, h, w)

    return res

Backward_tensorGrid = {}
def warp(tensorInput, tensorFlow, mode='bilinear', padding_mode = 'zeros', align_corners=False):
    if str(tensorFlow.size()[2:]) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(-1, -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(-1, -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size()[2:])] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(torch.device('cuda'), non_blocking = True)

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size()[2:])] + tensorFlow).permute(0, 2, 3, 1), mode=mode, padding_mode=padding_mode, align_corners=align_corners)