import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np

from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmedit.models.common import PixelShufflePack, ResidualBlockNoBN, make_layer
from mmedit.utils import get_root_logger
from models.archs.edvr_net import PCDAlignment, TSAFusion

from models.archs.SPyNet import SPyNet
from models.archs.RefVSR_.attention import AlignedAttention, FeatureMatching
from models.archs.RefVSR_.common import default_conv, BasicBlock, ResList
from models.utils import norm_res_vis, warp


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.config = config
        self.scale = config.scale
        self.flag_HD_in = config.flag_HD_in
        num_blocks = config.num_blocks
        mid_channels = config.mid_channels
        self.mid_channels = mid_channels
        self.keyframe_stride = config.keyframe_stride
        self.padding = 2

        self.edvr = EDVRFeatureExtractor(
            num_frames=self.padding * 2 + 1,
            center_frame_idx=self.padding,
            # pretrained='./ckpt/edvr.pytorch')
            pretrained='https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_reds_20210413-3867262f.pth')

        # self.FlowNet = SPyNet(pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth').to(torch.device('cuda'))
        self.FlowNet = SPyNet(pretrained='./ckpt/SPyNet.pytorch').to(torch.device('cuda'))
        for name, param in self.FlowNet.named_parameters():
            param.requires_grad = False

        self.feature_match = FeatureMatching(scale=self.scale, stride=1, flag_HD_in=self.flag_HD_in)
        ## AlignedAttention
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

        self.backward_fusion = nn.Conv2d(
            64 + mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.forward_fusion = nn.Conv2d(
            64 + mid_channels, mid_channels, 3, 1, 1, bias=True)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            2 * mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion_UP = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        if self.scale == 4:
            self.upsample2 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.forward_feat_prop_prev = None
        self.forward_flow_prev = None
        self.forward_feat_prop_UP_prev = None
        self.forward_conf_map_prop_prev = None

    def compute_up(self, backward_feat_UP, forward_feat_UP, conf_map_backward, conf_map_forward, base):
        ##
        conf_map_backward = F.interpolate(conf_map_backward, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        conf_map_forward = F.interpolate(conf_map_forward, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)


        cat_features = torch.cat([backward_feat_UP, forward_feat_UP], dim=1)
        out = self.fusion_UP(cat_features)
        alpha = self.conf_fusion_BWFW(torch.cat([conf_map_backward, conf_map_forward], dim=1))
        out = out + alpha * self.feat_fusion_BWFW(cat_features)
        out = self.feat_decoder_BWFW(out)
        ##

        if self.scale == 4:
            out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))

        out = self.conv_last(out) + base
        return out

    def AA_AF_conf_prop(self, lr, ref, conf_map, conf_map_prop, index_map, feat_prop, feat_prop_UP, ref_feat_down, ref_feat):
        lr_down = F.interpolate(lr, scale_factor=1/2, mode='bicubic', align_corners=False).clamp(0, 1)
        ref_feat_aligned = self.aa1(lr_down, ref, index_map, ref_feat_down, 'aa1')
        cat_features = torch.cat([feat_prop, ref_feat_aligned], dim=1)
        alpha = self.conf_fusion(torch.cat([conf_map_prop, conf_map], dim=1))
        feat_prop = feat_prop + alpha * self.feat_fusion(cat_features)
        feat_prop = self.feat_decoder(feat_prop)

        feat_prop_UP = self.feat_fusion2_1(torch.cat([feat_prop_UP, self.upsample1(feat_prop)], dim=1))
        ref_feat_aligned_UP = self.aa2(lr, ref, index_map, ref_feat, 'aa2')
        cat_features = torch.cat([feat_prop_UP, ref_feat_aligned_UP], dim=1)
        conf_map_prop_UP = F.interpolate(conf_map_prop, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        conf_map_UP = F.interpolate(conf_map, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)
        alpha = self.conf_fusion2(torch.cat([conf_map_prop_UP, conf_map_UP], dim=1))
        feat_prop_UP = feat_prop_UP + alpha * self.feat_fusion2(cat_features)
        feat_prop_UP = self.feat_decoder2(feat_prop_UP)

        conf_map_prop, _ = torch.max(torch.cat([conf_map_prop, conf_map], dim=1), dim=1, keepdim=True)

        return feat_prop, feat_prop_UP, conf_map_prop

    def spatial_padding(self, lrs):
        """ Apply pdding spatially.
        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = lrs.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        lrs = lrs.reshape(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.view(n, t, c, h + pad_h, w + pad_w)

    def compute_refill_features(self, lrs, keyframe_idx, h, w):
        """ Compute keyframe features for information-refill.
        Since EDVR-M is used, padding is performed before feature computation.
        Args:
            lrs (Tensor): Input LR images with shape (n, t, c, h, w)
            keyframe_idx (list(int)): The indices specifying the keyframes.
        Return:
            dict(Tensor): The keyframe features. Each key corresponds to the
                indices in keyframe_idx.
        """

        if self.padding == 2:
            lrs = [lrs[:, [4, 3]], lrs, lrs[:, [-4, -5]]]  # padding
        elif self.padding == 3:
            lrs = [lrs[:, [6, 5, 4]], lrs, lrs[:, [-5, -6, -7]]]  # padding
        lrs = torch.cat(lrs, dim=1)

        num_frames = 2 * self.padding + 1
        feats_refill = {}
        for i in keyframe_idx:
            feats_refill[i] = self.edvr(lrs[:, i:i + num_frames].contiguous())[:, :, :h, :w]
        return feats_refill

    def forward(self, lrs, refs, is_first_frame, is_log=False, is_train=False):
        """Forward function for BasicVSR.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Output HR (SR result of the center LR frame) with shape (n, c, 4h, 4w).
        """
        outs = collections.OrderedDict()
        if is_log:
            outs['vis'] = collections.OrderedDict()


        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        ## Computing forward & backward flows
        with torch.no_grad():
            forward_flows = []
            backward_flows = []
            for j in range(0, t-1):
                forward_flows.append(F.interpolate(self.FlowNet(lrs[:, j+1], lrs[:, j]), size=(h, w), mode='bilinear', align_corners=False)[:, None])
            for j in range(t-1, 0, -1):
                backward_flows.insert(0, F.interpolate(self.FlowNet(lrs[:, j-1], lrs[:, j]), size=(h, w), mode='bilinear', align_corners=False)[:, None])
            forward_flows = torch.cat(forward_flows, dim=1)
            backward_flows = torch.cat(backward_flows, dim=1)

        lrs = self.spatial_padding(lrs)
        # get the keyframe indices for information-refill
        if is_first_frame:
            self.keyframe_idx = np.arange(0, t, self.keyframe_stride)
        else:
            new_ki = self.keyframe_idx - 1
            new_ki = new_ki[new_ki>=0]
            self.keyframe_idx = np.arange(new_ki[0], t, self.keyframe_stride)

        if self.keyframe_idx[-1] != t - 1:
            self.keyframe_idx = np.append(self.keyframe_idx, t - 1)  # the last frame must be a keyframe

        # backward-time propgation
        feats_refill = self.compute_refill_features(lrs, self.keyframe_idx, h, w)
        lrs = lrs[:, :, :, :h, :w]

        ## Computing index_map and confidec between LR and Ref
        conf_maps = []
        index_maps = []
        for i in range(0, t):
            conf_map, index_map = self.feature_match(lrs[:, i], refs[:, i])

            conf_maps.append(conf_map)
            index_maps.append(index_map)

        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        feat_prop_UP = lrs.new_zeros(n, self.mid_channels, h*2, w*2)
        conf_map_prop = lrs.new_zeros(n, 1, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = backward_flows[:, i]
                feat_prop = warp(feat_prop, flow)
                conf_map_prop = warp(conf_map_prop, flow)
                feat_prop_UP = warp(feat_prop_UP, F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)

            if i in self.keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)

            ## RAP
            conf_map = conf_maps[i]
            index_map = index_maps[i]
            ref_feat = self.res1(self.ref_encoder1(refs[:, i])) # keeps scale
            ref_feat_down = self.res2(self.ref_encoder2(ref_feat)) # downscales
            feat_prop, feat_prop_UP, conf_map_prop = self.AA_AF_conf_prop(lrs[:, i], refs[:, i], conf_map, conf_map_prop, index_map, self.backward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1)), feat_prop_UP, ref_feat_down, ref_feat)
            ##
            if i == t//2:
                backward_feat_UP = feat_prop_UP
                conf_map_prop_backward = conf_map_prop

            outputs.append(feat_prop)

        outputs = outputs[::-1]
        if is_first_frame:
            feat_prop = torch.zeros_like(feat_prop)
            feat_prop_UP = torch.zeros_like(backward_feat_UP)
            conf_map_prop = torch.zeros_like(conf_map)

        for i in range(0, t//2+1):
            lr_curr = lrs[:, i]
            if i > 0: # no warping required for the first timestep
                feat_prop = warp(feat_prop, forward_flows[:, i-1])
                feat_prop_UP = warp(feat_prop, F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
                conf_map_prop = warp(conf_map_prop, flow)
            elif i == 0 and is_first_frame is False: # warp if the first feat_prop is computed for previous temporal frames
                feat_prop = warp(self.forward_feat_prop_prev, self.forward_flow_prev)
                feat_prop_UP = warp(self.forward_feat_prop_UP_prev, F.interpolate(input=self.forward_flow_prev, scale_factor=2, mode='bilinear', align_corners=True) * 2.0)
                conf_map_prop = warp(self.forward_conf_map_prop_prev, self.forward_flow_prev)

            if i in self.keyframe_idx: # information-refill
                feat_prop = torch.cat([feat_prop, feats_refill[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            conf_map = conf_maps[i]
            index_map = index_maps[i]
            ref_feat = self.res1(self.ref_encoder1(refs[:, i])) # keeps scale
            ref_feat_down = self.res2(self.ref_encoder2(ref_feat)) # downscales
            feat_prop, feat_prop_UP, conf_map_prop = self.AA_AF_conf_prop(lrs[:, i], refs[:, i], conf_map, conf_map_prop, index_map, self.forward_resblocks(torch.cat([lr_curr, outputs[i], feat_prop], dim=1)), feat_prop_UP, ref_feat_down, ref_feat)

            if i == 0:
                self.forward_feat_prop_prev = feat_prop.clone().detach()
                self.forward_flow_prev = forward_flows[:, i, :, :, :].clone().detach()
                self.forward_feat_prop_UP_prev = feat_prop_UP.clone().detach()
                self.forward_conf_map_prop_prev = conf_map_prop.clone().detach()

        base = F.interpolate(lrs[:, t//2], scale_factor=self.scale, mode='bicubic', align_corners=False).clamp(0, 1)
        out = self.compute_up(backward_feat_UP, feat_prop_UP, conf_map_prop_backward, conf_map_prop, base)

        if is_train is False:
            out = out.clamp(0, 1)
        outs['result'] = out

        if is_log and self.config.save_sample:
            with torch.no_grad():
                lr_down = F.interpolate(lrs[:, t//2], scale_factor=1/2, mode='bicubic', align_corners=False).clamp(0, 1)
                conf_map_prop_forward = conf_map_prop
                conf_map_prop, _ = torch.max(torch.cat([conf_map_prop_backward, conf_map_prop_forward], dim=1), dim=1, keepdim=True)
                ref_downsampled = F.interpolate(refs[:, t//2], scale_factor=1/2, mode='bicubic', align_corners=False).clamp(0, 1)
                outs['vis']['FW_aa1_fm_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa1(lr_down, refs[:, t//2], index_map, ref_downsampled, 'aa1', return_fm=True).clone().detach()
                if self.aa1.align:
                    outs['vis']['FW_aa1_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa1(lr_down, refs[:, t//2], index_map, ref_downsampled, 'aa1').clone().detach()
                outs['vis']['FW_aa2_fm_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa2(lrs[:, t//2], refs[:, t//2], index_map, refs[:, t//2], 'aa2', return_fm=True).clone().detach()
                if self.aa2.align:
                    outs['vis']['FW_aa2_ref_aligned{}'.format('' if not self.flag_HD_in else '')] = self.aa2(lrs[:, t//2], refs[:, t//2], index_map, refs[:, t//2], 'aa2').clone().detach()

                outs['vis']['conf_map_norm'] = norm_res_vis(conf_map)
                outs['vis']['conf_map_prop_backward_norm'] = norm_res_vis(conf_map_prop_backward)
                outs['vis']['conf_map_prop_forward_norm'] = norm_res_vis(conf_map_prop_forward)
                outs['vis']['conf_map_prop_norm'] = norm_res_vis(conf_map_prop)

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

class EDVRFeatureExtractor(nn.Module):
    """EDVR feature extractor for information-refill in IconVSR.
    We use EDVR-M in IconVSR. To adopt pretrained models, please
    specify "pretrained".
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
        pretrained (str): The pretrained model path. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 out_channel=3,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 with_tsa=True,
                 pretrained=None):

        super().__init__()

        self.center_frame_idx = center_frame_idx
        self.with_tsa = with_tsa
        act_cfg = dict(type='LeakyReLU', negative_slope=0.1)

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.feature_extraction = make_layer(
            ResidualBlockNoBN,
            num_blocks_extraction,
            mid_channels=mid_channels)

        # generate pyramid features
        self.feat_l2_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l2_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.feat_l3_conv1 = ConvModule(
            mid_channels, mid_channels, 3, 2, 1, act_cfg=act_cfg)
        self.feat_l3_conv2 = ConvModule(
            mid_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        # pcd alignment
        self.pcd_alignment = PCDAlignment(
            mid_channels=mid_channels, deform_groups=deform_groups)
        # fusion
        if self.with_tsa:
            self.fusion = TSAFusion(
                mid_channels=mid_channels,
                num_frames=num_frames,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frames * mid_channels, mid_channels, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

    def forward(self, x):
        """Forward function for EDVRFeatureExtractor.
        Args:
            x (Tensor): Input tensor with shape (n, t, 3, h, w).
        Returns:
            Tensor: Intermediate feature with shape (n, mid_channels, h, w).
        """

        n, t, c, h, w = x.size()

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        l1_feat = self.feature_extraction(l1_feat)
        # L2
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        # L3
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)

        # pcd alignment
        ref_feats = [  # reference feature list
            l1_feat[:, self.center_frame_idx].clone(),
            l2_feat[:, self.center_frame_idx].clone(),
            l3_feat[:, self.center_frame_idx].clone()
        ]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [
                l1_feat[:, i].clone(), l2_feat[:, i].clone(),
                l3_feat[:, i].clone()
            ]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)

        return feat
