import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import gc

from mmedit.models.common import PixelShufflePack, ResidualBlockNoBN, make_layer

from models.archs.SPyNet import SPyNet
from models.archs.RefVSR_.attention import AlignedAttention, FeatureMatching
from models.archs.RefVSR_.common import default_conv, BasicBlock, ResList
from models.utils import norm_res_vis, warp

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.config = config
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.scale = config.scale
        self.flag_HD_in = config.flag_HD_in
        num_blocks = config.num_blocks
        mid_channels = config.mid_channels
        self.mid_channels = mid_channels

        # self.FlowNet = SPyNet(pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth').to(torch.device(self.config.device))
        self.FlowNet = SPyNet(pretrained='./ckpt/SPyNet.pytorch', device=self.config.device)
        for name, param in self.FlowNet.named_parameters():
            param.requires_grad = False

        self.feature_match = FeatureMatching(scale=self.scale, stride=1, flag_HD_in=self.flag_HD_in)
        ## AlignedAttention (DCSR, ICCV2021)
            # - scale: kernel size and stride for deform conv layer
            # (basically local patch size to align)
            # => 2*size of LRxX / size of index map (size of LRx4)
            # => In case of x4, it is 2 * (LRx4/LRx2)=2. In case of x2, it is 2 * (LRx2/LRx4)=4
            # - align: whether to align using deform conv (affine - rotation and scaling)
            # (if false, warping based on cosine-similarity feature matching is applied)
        self.aa1 = AlignedAttention(scale=config.matching_ksize//2, align=True if config.matching_ksize//2 > 1 else False)
        self.aa2 = AlignedAttention(scale=config.matching_ksize, align=True)

        m_head1 = [BasicBlock(default_conv, 3, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        m_head2 = [BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=2, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]

        conf_fusion = [BasicBlock(default_conv, 2, 16, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, 16, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        conf_fusion2 = [BasicBlock(default_conv, 2, 16, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, 16, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        conf_fusion_BWFW = [BasicBlock(default_conv, 2, 16, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, 16, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]

        feat_fusion = [BasicBlock(default_conv, 2*mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        feat_fusion2_1 = [BasicBlock(default_conv, 2*mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        feat_fusion2 = [BasicBlock(default_conv, 2*mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]
        feat_fusion_BWFW = [BasicBlock(default_conv, 2*mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True)),
                   BasicBlock(default_conv, mid_channels, mid_channels, 3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True))]

        self.ref_encoder1 = nn.Sequential(*m_head1)
        self.res1 = ResList(4, mid_channels)
        self.ref_encoder2 = nn.Sequential(*m_head2)
        self.res2 = ResList(4, mid_channels)

        self.conf_fusion = nn.Sequential(*conf_fusion)
        self.feat_fusion = nn.Sequential(*feat_fusion)
        self.feat_decoder = ResList(8, mid_channels)

        self.conf_fusion2 = nn.Sequential(*conf_fusion2)
        self.feat_fusion2_1 = nn.Sequential(*feat_fusion2_1)
        self.feat_fusion2 = nn.Sequential(*feat_fusion2)
        self.feat_decoder2 = ResList(4, mid_channels)

        self.conf_fusion_BWFW = nn.Sequential(*conf_fusion_BWFW)
        self.feat_fusion_BWFW = nn.Sequential(*feat_fusion_BWFW)
        self.feat_decoder_BWFW = ResList(4, mid_channels)

        ## propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            1 * mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            1 * mid_channels + 3, mid_channels, num_blocks)

        ## upsample
        self.fusion_UP = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        if self.scale == 4:
            self.upsample2 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.forward_feat_prop_prev = None
        self.forward_flow_prev = None
        self.forward_feat_prop_UP_prev = None
        self.forward_conf_map_prop_prev = None
        self.frame_itr_num = 0
        self.max_frame_itr_num = self.config.reset_branch

    ## Upsampling module
    def compute_up(self, backward_feat_UP, forward_feat_UP, conf_map_backward, conf_map_forward, base):
        conf_map_backward = F.interpolate(conf_map_backward, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        conf_map_forward = F.interpolate(conf_map_forward, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)

        cat_features = torch.cat([backward_feat_UP, forward_feat_UP], dim=1)
        out = self.fusion_UP(cat_features)
        alpha = self.conf_fusion_BWFW(torch.cat([conf_map_backward, conf_map_forward], dim=1))
        out = out + alpha * self.feat_fusion_BWFW(cat_features)
        out = self.feat_decoder_BWFW(out)

        if self.scale == 4:
            out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))

        out = self.conv_last(out) + base
        return out

    ## The Reference Alignment and Propagation (RAP) module (Fig. 4 of the main paper).
    ## - The paper explains the simplified version of RAP module, here, we use 2-level RAP
    def AA_AF_conf_prop(self, lr, ref, conf_map, conf_map_prop, index_map, feat_prop, feat_prop_UP, ref_feat_down, ref_feat):
        ## the first level of the RAP module
        lr_down = F.interpolate(lr, scale_factor=1/2, mode='bicubic', align_corners=False).clamp(0, 1)
        # reference alignment
        ref_feat_aligned = self.aa1(lr_down, ref, index_map, ref_feat_down, 'aa1')
        # propagative temporal fusion
        cat_features = torch.cat([feat_prop, ref_feat_aligned], dim=1)
        alpha = self.conf_fusion(torch.cat([conf_map_prop, conf_map], dim=1))
        feat_prop = feat_prop + alpha * self.feat_fusion(cat_features)
        feat_prop = self.feat_decoder(feat_prop)

        ## the second level of the RAP module
        # reference alignment
        ref_feat_aligned_UP = self.aa2(lr, ref, index_map, ref_feat, 'aa2')
        # propagative temporal fusion
        feat_prop_UP = self.feat_fusion2_1(torch.cat([feat_prop_UP, self.upsample1(feat_prop)], dim=1)) # aggregates upampled features of the preovious recurrent step
        cat_features = torch.cat([feat_prop_UP, ref_feat_aligned_UP], dim=1)
        conf_map_prop_UP = F.interpolate(conf_map_prop, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        conf_map_UP = F.interpolate(conf_map, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        alpha = self.conf_fusion2(torch.cat([conf_map_prop_UP, conf_map_UP], dim=1))
        feat_prop_UP = feat_prop_UP + alpha * self.feat_fusion2(cat_features)
        feat_prop_UP = self.feat_decoder2(feat_prop_UP)

        # confidence accumulation
        conf_map_prop, _ = torch.max(torch.cat([conf_map_prop, conf_map], dim=1), dim=1, keepdim=True)

        return feat_prop, feat_prop_UP, conf_map_prop

    def forward(self, lrs, refs, is_first_frame, is_log=False, is_train=False):
        """Forward function for RefVSR.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
            refs (Tensor): Input Ref sequence with shape (n, t, c, h, w). None: Spatial resolution of the tensors are twice as the LR during training, equal during validation and evaluation.
            is_first_frame (boolean): whether lrs[:, 0] and refs[:, 0] are the first frame of a video sequence.
            is_log (boolean): whether to return samples (e.g., confidence map, warppred image, etc. for debugging. The return is enabled when config.save_sample is True.)
            is_train (boolean): whether it is trainig phase.
        Returns:
            Tensor: Output HR (SR result of the center LR frame) with shape (n, c, 4h, 4w).
        """
        outs = collections.OrderedDict()
        if is_log:
            outs['vis'] = collections.OrderedDict()
        n, t, c, h, w = lrs.size()

        # enable is_first_frame True when number of iterations passed the pre-set limit (= resests features propagation of the forward branch).
        if is_train == False:
            if self.max_frame_itr_num is not None and self.frame_itr_num == self.max_frame_itr_num:
                is_first_frame = True

        # if is_first_frame is True, forward branches processes frames from t=0, otherwise, uses saved features from the previous iteration computes only the center frame (t//2).
        if is_first_frame:
            range_start = 0
        else:
            range_start = t//2 if is_train is False else 0

        # computing forward & backward flows (S in Fig. 3 of the paper)
        with torch.no_grad():
            forward_flows = []
            backward_flows = []
            for j in range(0, t-1):
                forward_flows.append(F.interpolate(self.FlowNet(lrs[:, j+1], lrs[:, j]), size=(h, w), mode='bilinear', align_corners=False)[:, None])
            for j in range(t-1, 0, -1):
                backward_flows.insert(0, F.interpolate(self.FlowNet(lrs[:, j-1], lrs[:, j]), size=(h, w), mode='bilinear', align_corners=False)[:, None])
            forward_flows = torch.cat(forward_flows, dim=1)
            backward_flows = torch.cat(backward_flows, dim=1)

        # computing index_map and confidec between LR and Ref (cosine similarity part in the RAP module in Fig. 4 of the main paper)
        conf_maps = []
        index_maps = []
        for i in range(0, t):
            if i >= range_start:
                conf_map, index_map = self.feature_match(lrs[:, i], refs[:, i])
            else:
                conf_map, index_map = None, None
            conf_maps.append(conf_map)
            index_maps.append(index_map)

        if is_train is False:
            gc.collect()
            torch.cuda.empty_cache()

        ## BACKWARD BRANCH ##
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_UP = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        conf_map_prop = lrs.new_zeros(n, 1, h, w)
        for i in range(t - 1, t//2 - 1, -1):
            ## inter-frame alignment (warp in Fig. 3 of the main paper)
            if i < t - 1:  # no warping required for the last timestep
                flow = backward_flows[:, i]
                feat_prop = warp(feat_prop, flow)
                conf_map_prop = warp(conf_map_prop, flow)
                feat_prop_UP = warp(feat_prop_UP, F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
                if is_log and i == t//2:
                    outs['vis']['BW_LR_next_warp'] = warp(lrs[:, i+1], flow).detach().clone()

            ## feature aggregation (R in Fig. 3 of the main paper)
            feat_prop = torch.cat([lrs[:, i], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            ## The Reference Alignment and Propagation (RAP) module (Fig. 4 in the main paper)
            # pre-computed matching confidence and mathcing index
            conf_map = conf_maps[i]
            index_map = index_maps[i]
            # reference alignment and propagative temporal fusion
            ref_feat = self.res1(self.ref_encoder1(refs[:, i])) # keeps scale
            ref_feat_down = self.res2(self.ref_encoder2(ref_feat)) # downscales
            feat_prop, feat_prop_UP, conf_map_prop = self.AA_AF_conf_prop(lrs[:, i], refs[:, i], conf_map, conf_map_prop, index_map, feat_prop, feat_prop_UP, ref_feat_down, ref_feat)

        backward_feat_UP = feat_prop_UP
        conf_map_prop_backward = conf_map_prop

        ## FORWARD BRANCH ##
        if is_first_frame:
            feat_prop = torch.zeros_like(feat_prop)
            feat_prop_UP = torch.zeros_like(backward_feat_UP)
            conf_map_prop = torch.zeros_like(conf_map)
            range_start = 0
        else:
            range_start = t//2 if is_train is False else 0

        for i in range(range_start, t//2+1):
            ## inter-frame alignment (warp in Fig. 3 of the main paper)
            if i > range_start: # no warping required for the first timestep
                flow = forward_flows[:, i-1, :, :, :]
                feat_prop = warp(feat_prop, flow)
                feat_prop_UP = warp(feat_prop, F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
                conf_map_prop = warp(conf_map_prop, flow)
            elif i == range_start and is_first_frame is False: # warp if the first feat_prop is computed for previous temporal frames
                flow = self.forward_flow_prev
                feat_prop = warp(self.forward_feat_prop_prev, flow)
                feat_prop_UP = warp(self.forward_feat_prop_UP_prev, F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
                conf_map_prop = warp(self.forward_conf_map_prop_prev, flow)

            if is_log and i == t//2:
                outs['vis']['FW_LR_prev_warp'] = warp(lrs[:, i-1], flow).detach().clone()

            ## feature aggregation (R in Fig. 3 of the main paper)
            feat_prop = torch.cat([lrs[:, i], feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            ## The Reference Alignment and Propagation (RAP) module (Fig. 4 in the main paper)
            # pre-computed matching confidence and mathcing index
            conf_map = conf_maps[i]
            index_map = index_maps[i]
            # reference alignment and propagative temporal fusion
            ref_feat = self.res1(self.ref_encoder1(refs[:, i])) # keeps scale
            ref_feat_down = self.res2(self.ref_encoder2(ref_feat)) # downscales
            feat_prop, feat_prop_UP, conf_map_prop = self.AA_AF_conf_prop(lrs[:, i], refs[:, i], conf_map, conf_map_prop, index_map, feat_prop, feat_prop_UP, ref_feat_down, ref_feat)

            # keep features, flow, confidence maps for the next iteration to compute only the center frame.
            if (is_train and i == 0) or (is_train is False and i == t//2):
                self.forward_feat_prop_prev = feat_prop.detach().clone()
                self.forward_flow_prev = forward_flows[:, i, :, :, :].detach().clone()
                self.forward_feat_prop_UP_prev = feat_prop_UP.detach().clone()
                self.forward_conf_map_prop_prev = conf_map_prop.detach().clone()

        ## U in Fig. 2 of the main paper
        forward_feat_UP = feat_prop_UP
        conf_map_prop_forward = conf_map_prop
        base = F.interpolate(lrs[:, t//2], scale_factor=self.scale, mode='bicubic', align_corners=False).clamp(0, 1)
        out = self.compute_up(backward_feat_UP, forward_feat_UP, conf_map_prop_backward, conf_map_prop_forward, base)

        # reset iteration count if is_first_frame is True
        if is_train is False:
            if is_first_frame:
                self.frame_itr_num = 0
            self.frame_itr_num += 1

            out = out.clamp(0, 1)
        outs['result'] = out

        ################################## Debugging Samples ##################################
        if is_log and self.config.save_sample:
            with torch.no_grad():
                lr_down = F.interpolate(lrs[:, t//2], scale_factor=1/2, mode='bicubic', align_corners=False).clamp(0, 1)
                conf_map_prop, _ = torch.max(torch.cat([conf_map_prop_backward, conf_map_prop_forward], dim=1), dim=1, keepdim=True)
                ref_downsampled = F.interpolate(refs[:, t//2], scale_factor=1/2, mode='bicubic', align_corners=False).clamp(0, 1)
                outs['vis']['FW_aa1_fm_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa1(lr_down, refs[:, t//2], index_map, ref_downsampled, 'aa1', return_fm=True).detach().clone()
                if self.aa1.align:
                    outs['vis']['FW_aa1_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa1(lr_down, refs[:, t//2], index_map, ref_downsampled, 'aa1').detach().clone()
                outs['vis']['FW_aa2_fm_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa2(lrs[:, t//2], refs[:, t//2], index_map, refs[:, t//2], 'aa2', return_fm=True).detach().clone()
                if self.aa2.align:
                    outs['vis']['FW_aa2_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa2(lrs[:, t//2], refs[:, t//2], index_map, refs[:, t//2], 'aa2').detach().clone()

                outs['vis']['conf_map_norm'] = norm_res_vis(conf_map)
                outs['vis']['conf_map_prop_backward_norm'] = norm_res_vis(conf_map_prop_backward)
                outs['vis']['conf_map_prop_forward_norm'] = norm_res_vis(conf_map_prop_forward)
                outs['vis']['conf_map_prop_norm'] = norm_res_vis(conf_map_prop)

                outs['eval_vis'] = collections.OrderedDict()
                outs['eval_vis']['conf_map'] = conf_map
                outs['eval_vis']['conf_map_prop'] = conf_map_prop
                outs['eval_vis']['conf_map_prop_backward'] = conf_map_prop_backward
                outs['eval_vis']['conf_map_prop_forward'] = conf_map_prop_forward
        #######################################################################################

        return outs

class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)
