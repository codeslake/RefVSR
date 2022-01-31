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
    res = res.clone().detach()
    b, c, h, w = res.size()

    res = res.view(res.size(0), -1)
    res = res - res.min(1, keepdim=True)[0]
    res = res / res.max(1, keepdim=True)[0]
    res = res.view(b, c, h, w)

    return res

def OF_vis(OF):
    OF = OF.cpu().numpy().transpose(1, 2, 0)
    OF = flow2img(OF)
    OF = torch.FloatTensor(np.expand_dims(OF.transpose(2, 0, 1), axis = 0))
    return OF

#from DPDD code
def get_psnr2(img1, img2, PIXEL_MAX=1.0):
    mse_ = torch.mean( (img1 - img2) ** 2 )
    return 10 * torch.log10(PIXEL_MAX / mse_)

    # return calculate_psnr(img1, img2)

Backward_tensorGrid = {}
def warp(tensorInput, tensorFlow, mode='bilinear', padding_mode = 'zeros', align_corners=False):
    if str(tensorFlow.size()[2:]) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(-1, -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(-1, -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size()[2:])] = torch.cat([ tensorHorizontal, tensorVertical ], 1).to(torch.device('cuda'), non_blocking = True)

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
#
    # print(tensorFlow.size())
    # print(Backward_tensorGrid[str(tensorFlow.size()[2:])].size())
    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size()[2:])] + tensorFlow).permute(0, 2, 3, 1), mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def FM(F1, F2, F3, corr_, pool=None, scale = None, align_corners=False):
    # rank = torch.distributed.get_rank()
    shape = F1.size()
    channel = shape[1]
    # l2norm = torch.sqrt(torch.sum(torch.mul(F1,F1),1,keepdim=True) + 1e-8)
    # l2norm2 = torch.sqrt(torch.sum(torch.mul(F2,F2),1,keepdim=True) + 1e-8)
    F1 = F.normalize(F1, p = 2, dim=1)
    F2 = F.normalize(F2, p = 2, dim=1)
    corr = channel * corr_(F1, F2)

    if pool is not None:
        corr = pool(corr)

    with torch.no_grad():
        matching_index = torch.argmax(corr, dim=1).type(torch.cuda.FloatTensor)

        kernel_size = corr_.max_displacement * 2 + 1
        half_ks = np.floor(kernel_size/(corr_.stride2*2))
        y = ((matching_index//np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2
        x = ((matching_index%np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2

        flow = torch.cat((torch.unsqueeze(x,1),torch.unsqueeze(y,1)), axis=1)
        if scale is not None:
            flow = F.interpolate(input=flow, scale_factor=scale, mode='nearest') * scale

        n = warp(F3, flow, 'nearest', align_corners=align_corners)

    return n, flow

def unfold_corr_topk(F1, F2, F3, corr_, topk, pool=None, scale = None):
    # rank = torch.distributed.get_rank()
    b, c, h, w = F1.size()
    F1 = F.normalize(F1, p = 2, dim=1)
    F2 = F.normalize(F2, p = 2, dim=1)
    corr = c * corr_(F1, F2)

    if pool is not None:
        corr = pool(corr)

    _, matching_index_topk = torch.topk(corr, topk, dim=1)
    matching_index_topk = matching_index_topk.unsqueeze_(dim=2).repeat(1, 1, c, 1, 1)

    kernel_size = corr_.max_displacement * 2 + 1
    F3 = F.unfold(F3, kernel_size, padding = kernel_size // 2).view(b, -1, kernel_size**2, h, w).permute(0, 2, 1, 3, 4)
    n = torch.gather(F3, 1, matching_index_topk)

    return n

def FM2(F1, F2, F3, corr_, scale = None):
    # rank = torch.distributed.get_rank()

    channel = F1.size()[1]
    F1 = F.normalize(F1, p=2, dim=1)
    F2 = F.normalize(F2, p=2, dim=1)
    corr = channel * corr_(F1, F2)

    ## TODO: corr modulation

    soft, matching_index = torch.max(corr, dim=1, keepdim = True)

    b, c, h, w = soft.size()
    unfold_k = 5
    disp_val = F.unfold(soft, kernel_size=unfold_k, padding=unfold_k//2).view(b, c*unfold_k**2, h, w)
    disp_val, disp_idx  = torch.max(disp_val, dim=1, keepdim = True)

    kernel_size = unfold_k
    half_ks = np.floor(kernel_size/(1*2))
    y = ((disp_idx //np.floor(kernel_size+0.5)) - half_ks)
    x = ((disp_idx %np.floor(kernel_size+0.5)) - half_ks)
    disp_map = torch.cat([x,y], axis=1) # displacement map indicating neighbor coordinate with max correlation value

    kernel_size = corr_.max_displacement * 2 + 1
    half_ks = np.floor(kernel_size/(corr_.stride2*2))
    y = ((matching_index//np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2
    x = ((matching_index%np.floor(kernel_size/float(corr_.stride2)+0.5)) - half_ks) * corr_.stride2

    flow = torch.cat([x,y], axis=1)
    flow = warp(flow, disp_map, mode='nearest')

    if scale is not None:
        flow = F.interpolate(input=flow, scale_factor=scale, mode='nearest') * scale
    n = warp(F3, flow, mode='nearest')

    return n, flow

grids = {}
def get_OF_grid(size):
    if str(size) not in grids:
        y_position = torch.ones(size).cumsum(0).float()[None, None, :]
        x_position = torch.ones(size).cumsum(1).float()[None, None, :]
        grids[str(size)] = torch.cat([x_position, y_position], dim=1).to(torch.device('cuda')) - 1
        # grids[str(size)] = torch.cat([x_position, y_position], dim=1) - 1

    return grids[str(size)]