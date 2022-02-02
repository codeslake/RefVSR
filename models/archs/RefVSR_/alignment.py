###############################################################################################
## The code is adopted from https://github.com/Tengfei-Wang/DCSR/blob/master/model/attention.py
###############################################################################################

import torch
from torch import nn
import torch.nn.functional as F
from models.archs.RefVSR_.common import *

class AlignedConv2d(nn.Module):
    def __init__(self, inc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(AlignedConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ReflectionPad2d(padding)

        head = [nn.Conv2d(3, 32, kernel_size=5, padding=2, stride=1), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True)]

        head2 = [nn.Conv2d(2*32, 32, kernel_size=5, padding=2, stride=stride), nn.LeakyReLU(0.2, inplace=True), ResBlock(in_channels=32, out_channels=32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1)]

        self.p_conv = nn.Sequential(*head2)
        self.conv1 = nn.Sequential(*head)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.conv1.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(2*inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    #warpped_features_, lr, warpped_ref
    def forward(self, x, query, ref):
        # print('\n\twarped_features:', x.size(), '\n\tlr:', query.size(), '\n\twarpped_ref:', ref.size(), '\n\tis_lr_up:', is_query_up, '\n')
        query = F.interpolate(query, scale_factor=2, mode='bicubic', align_corners=False)
        query = self.conv1(query)
        ref = self.conv1(ref)

        affine = self.p_conv( torch.cat((ref, query), 1)) + 1.
        if self.modulation:
            m = torch.sigmoid(self.m_conv(torch.cat((ref, query), 1)))

        dtype = affine.data.type()
        ks = self.kernel_size
        N = ks*ks

        if self.padding:
            x = self.zero_padding(x)

        affine = torch.clamp(affine, -3, 3)
        # (b, 2N, h, w)
        p = self._get_p(affine, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        alignment = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(alignment.size(1))], dim=1)
            alignment *= m

        alignment = self._reshape_alignment(alignment, ks)

        return alignment

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-1*((self.kernel_size-1)//2)-0.5, (self.kernel_size-1)//2+0.6, 1.),  torch.arange(-1*((self.kernel_size-1)//2)-0.5, (self.kernel_size-1)//2+0.6, 1.))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, affine, dtype):
        N, h, w = self.kernel_size*self.kernel_size, affine.size(2), affine.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)

        p =  p_n.repeat(affine.size(0), 1, h, w)
        p = p.permute(0,2,3,1) #(1,  h, w, 2N)
        affine = affine.permute(0,2,3,1) #-1xh x w x 3

        s_x =  affine[:,:,:,0:1]
        s_y =  affine[:,:,:,1:2]

        p[:,:,:,:N] = p[:,:,:,:N].clone()*s_x.type(dtype)
        p[:,:,:,N:] = p[:,:,:,N:].clone()*s_y.type(dtype)
        p = p.view(p.shape[0],p.shape[1], p.shape[2], 1, p.shape[3]) #(1,  h, w, 1, 2N)
        p= torch.cat((p[:,:,:,:,:N], p[:,:,:,:,N:]), 3) #(1,  h, w, 2, N)
        p = p.permute(0,1,2,4,3) #(1,  h, w,  N, 2)

        theta = (affine[:,:,:,2:] - 1.)*1.0472
        rm = torch.cat((torch.cos(theta), torch.sin(theta),-1*torch.sin(theta), torch.cos(theta)), 3)
        rm = rm.view(affine.shape[0],affine.shape[1],affine.shape[2],2,2 ) #-1xh x w x 2x2
        result = torch.matmul(p,rm) #-1xh x w x Nx2
        result= torch.cat((result[:,:,:,:,0], result[:,:,:,:,1]), 3) #(-1,  h, w,  2N)

        # result = result.permute(0,3,1,2) +((self.kernel_size-1)//2-0.5) + p_0#-1, 2N, h, w
        result = result.permute(0,3,1,2) + (self.kernel_size-1)//2+0.5 + p_0#-1, 2N, h, w

        return result

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        padded_h = x.size(2)#
        c = x.size(1)
        # (b, c, h*w )
        # x_ = x#
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # print('\n\n2', padded_h*padded_w-1, x_.size(), q.size(), index.min(), index.max(), '\n\n')
        # index = torch.clamp(index, 0, padded_w*padded_h-1)
        # assert any(torch.isnan(index).flatten())==False

        result = x.gather(dim=-1, index=index.long()).contiguous().view(b, c, h, w, N)

        return result

    @staticmethod
    def _reshape_alignment(alignment, ks):
        b, c, h, w, N = alignment.size()
        alignment = torch.cat([alignment[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        alignment = alignment.contiguous().view(b, c, h*ks, w*ks)

        return alignment
