import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from models.archs.RefVSR_.common import *
from models.archs.RefVSR_.utils import *
from models.archs.RefVSR_.alignment import AlignedConv2d

class FeatureMatching(nn.Module):
    def __init__(self, scale=2, stride=1, flag_HD_in=False):
        super(FeatureMatching, self).__init__()
        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        self.ksize = 3

        self.scale = scale
        self.stride = stride
        self.flag_HD_in = flag_HD_in

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.feature_extract = torch.nn.Sequential()

         # VGG19: 0: conv (64) # 1: relu # 2: conv (64) # 3: relu # 4: maxpool # 5: conv (128) # 6: relu
        if self.flag_HD_in is False:
            vgg_range = 4 if self.scale == 4 else 7
        else:
            vgg_range = 7

        self.vgg_range = vgg_range
        for x in range(vgg_range):
            self.feature_extract.add_module(str(x), vgg_pretrained_features[x])

        match0 =  BasicBlock(default_conv, 64 if vgg_range == 4 else 128, 16, 1,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))
        self.feature_extract.add_module('map{}'.format(64 if vgg_range == 4 else 128), match0)

        for param in self.feature_extract.parameters():
            param.requires_grad = True
            # param.requires_grad = False

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 , 0.224, 0.225 )
        self.sub_mean = MeanShift(1, vgg_mean, vgg_std)
        self.avgpool = nn.AvgPool2d((2,2),(2,2))

        if self.flag_HD_in:
            self.scale_factor_x2 = 1/(self.scale//2)
        else:
            self.scale_factor_x2 = self.scale/2

    def forward(self, lr, ref, ref_downsample=True):
        h, w = lr.size()[2:]

        # if self.rank <= 0: print('\n\n1.', self.scale_factor_x2, lr.size(), ref.size())
        lr = self.sub_mean(lr)
        ref = self.sub_mean(ref)

        if self.flag_HD_in:
            lr  = F.interpolate(lr, scale_factor=self.scale_factor_x2, mode='nearest')
            ref  = F.interpolate(ref, scale_factor=self.scale_factor_x2, mode='nearest')

        # if self.rank <= 0: print('2.', self.scale_factor_x2, lr.size(), ref.size())
        lr_f = self.feature_extract(lr)

        lr_p = extract_image_patches(lr_f, ksizes=[self.ksize, self.ksize], strides=[self.stride,self.stride], rates=[1, 1], padding='same')

        if ref_downsample:
            ref_down = self.avgpool(ref)
        else:
            ref_down = ref

        # ref_down = F.interpolate(ref, scale_factor=1/2, mode='bicubic', align_corners=True).clamp(0, 1)
        ref_f = self.feature_extract(ref_down)
        ref_p = extract_image_patches(ref_f, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')

        ref_p = ref_p.permute(0, 2, 1)
        ref_p = F.normalize(ref_p, dim=2) # [N, Hr*Wr, C*k*k]
        lr_p  = F.normalize(lr_p, dim=1) # [N, C*k*k, H*W]

        # if self.rank <= 0: print('3.', self.scale_factor_x2, lr_f.size(), ref_f.size(), ref_down.size(), ref_p.size(), lr_p.size(), '\n\n')
        N, hrwr, _ = ref_p.size()
        _, _, hw = lr_p.size()
        # relavance_maps, hard_indices = torch.max(torch.bmm(ref_p, lr_p), dim=1) #[N, H*W]
        relavance_maps, hard_indices = torch.max(torch.einsum('bij,bjk->bik', ref_p.contiguous(), lr_p.contiguous()), dim=1) #[N, H*W]

        shape_lr = lr_f.shape
        relavance_maps = relavance_maps.view(shape_lr[0], 1, shape_lr[2], shape_lr[3])

        h_c, w_c = relavance_maps.size()[2:]
        if h/h_c != 1.:
            relavance_maps = F.interpolate(relavance_maps, scale_factor=h/h_c, mode='bicubic', align_corners=False).clamp(0, 1)

        return relavance_maps, hard_indices

class AlignedAttention(nn.Module):
    def __init__(self,  ksize=3, k_vsize=1, scale=1, stride=1, align=False):
        super(AlignedAttention, self).__init__()
        try:
            self.rank = torch.distributed.get_rank()
        except:
            self.rank = 0

        self.ksize = ksize
        self.k_vsize = k_vsize
        self.stride = stride
        self.scale = scale
        self.align = align
        if align:
          self.align = AlignedConv2d(inc=128, kernel_size=self.scale*self.k_vsize, padding=1, stride=self.scale*1, bias=None, modulation=False)

    def warp(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...] == [b, c, hw]
        # dim: scalar > 0
        # index: [N, idx] == [b, hw]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))] # [b, 1, -1]
        expanse = list(input.size()) # [b, c, hw]
        expanse[0] = -1 # [-1, c, hw]
        expanse[dim] = -1 # [-1, c, -1]
        index = index.view(views).expand(expanse) # [b, 1, hw] --> [b, c, hw]
        return torch.gather(input, dim, index) # [b, c, hw]

    #input_down, ref_p, index_map, ref_features2
    def forward(self, lr, ref, index_map, value, name, return_fm=False): #(1/2, 1, 1, 1/2)
        # value there can be features or image in ref view

        # b*c*h*w
        shape_out = list(lr.size())   # b*c*h*w

        # kernel size on input for matching
        kernel = self.scale*self.k_vsize

        # unfolded_value is extracted for reconstruction

        unfolded_value = extract_image_patches(value, ksizes=[kernel, kernel], strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L]
        warpped_value = self.warp(unfolded_value, 2, index_map)
        warpped_features_ = F.fold(warpped_value, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale)
        # if self.rank <= 0: print('[AlignedAttention]name: ', name, '\nlr: ', lr.size(), '\nref: ', ref.size(), '\nindex_map: ', index_map.size(), '\nindex_map.max(): ', index_map.max(), '\nvalue: ', value.size(), '\nunfolded_value: ', unfolded_value.size(), '\nscale: ', self.scale, '\nk_vsize: ', self.k_vsize, '\nkernel: ', kernel, '\nstride: ', self.stride, '\nwarpped_value: ', warpped_value.size(), '\nwarpped_features_: ', warpped_features_.size(), '\n')

        # for debugging
        if return_fm:
            return warpped_features_

        if self.align:
          unfolded_ref = extract_image_patches(ref, ksizes=[kernel, kernel],  strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L]
          warpped_ref = self.warp(unfolded_ref, 2, index_map)
          warpped_ref = F.fold(warpped_ref, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale)
          warpped_features = self.align(warpped_features_, lr, warpped_ref)
        else:
            return warpped_features_

        return warpped_features

class PatchSelect(nn.Module):
    def __init__(self,  stride=1):
        super(PatchSelect, self).__init__()
        self.stride = stride

    def forward(self, lr, ref):
        shape_lr = lr.shape
        shape_ref = ref.shape

        P = shape_ref[3] - shape_lr[3] + 1 #patch number per row
        ref = extract_image_patches(ref, ksizes=[shape_lr[2], shape_lr[3]], strides=[self.stride, self.stride], rates=[1, 1], padding='valid')

        lr = lr.view(shape_lr[0], shape_lr[1]* shape_lr[2] *shape_lr[3],1)

        y = torch.mean(torch.abs(ref - lr), 1)

        relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True) #[N, H*W]


        return  hard_indices.view(-1), P, relavance_maps

