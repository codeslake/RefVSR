import os
from importlib import import_module

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import toRed
from models.utils import norm_res_vis
import torch.distributed as dist

class Loss():
    def __init__(self, config):
        super(Loss, self).__init__()
        # print('Preparing loss function:')

        self.config = config
        self.is_train = config.is_train
        self.device = config.device
        self.loss = []
        self.scale = config.scale
        gaussian_module=import_module('models.loss.gaussian')
        self.gaussian_layer=getattr(gaussian_module, 'GaussianLayer')()

        self.rank = torch.distributed.get_rank() if config.dist else -1

        if self.is_train:
            for loss in config.loss.split('+'):
                if loss != '':
                    weight, loss_type = loss.split('*')
                    if loss_type == 'MSE':
                        loss_function = nn.MSELoss()

                    elif loss_type == 'L1':
                        loss_function = nn.L1Loss()

                    elif loss_type == 'L1_lf':
                        loss_function = nn.L1Loss()

                    elif loss_type in ['FID_hr', 'FID_ref', 'MFID_ref']:
                        module = import_module('models.loss.contextual')
                        loss_function = getattr(module, 'ContextualLoss')(vgg_layer=config.CX_vgg_layer).to(self.device)

                    elif loss_type in ['FID_hr_CoBi', 'FID_ref_CoBi','MFID_ref_CoBi']:
                        module = import_module('models.loss.contextual')
                        loss_function = getattr(module, 'ContextualLoss')(vgg_layer=config.CX_vgg_layer, is_CoBi=True).to(self.device)

                    elif loss_type in ['FID_ref_L2', 'MFID_ref_L2', 'FID_hr_L2']:
                        module = import_module('models.loss.contextual')
                        loss_function = getattr(module, 'ContextualLoss')(vgg_layer=config.CX_vgg_layer, band_width=0.5, loss_type='L2').to(self.device)

                    elif loss_type in ['FID_ref_L1', 'MFID_ref_L1', 'FID_hr_L1']:
                        module = import_module('models.loss.contextual')
                        loss_function = getattr(module, 'ContextualLoss')(vgg_layer=config.CX_vgg_layer, loss_type='L1').to(self.device)

                    elif loss_type in ['FID_ref_X_mu', 'MFID_ref_X_mu']:
                        module = import_module('models.loss.contextual_X_mu')
                        loss_function = getattr(module, 'ContextualLoss')(vgg_layer=config.CX_vgg_layer).to(self.device)

                    elif loss_type in ['FID_ref_CoBi_X_mu', 'MFID_ref_CoBi_X_mu']:
                        module = import_module('models.loss.contextual_X_mu')
                        loss_function = getattr(module, 'ContextualLoss')(vgg_layer=config.CX_vgg_layer, is_CoBi=True).to(self.device)

                    self.loss.append({
                            'type': loss_type,
                            'weight': float(weight),
                            'function': loss_function}
                    )

            self.gaussian_layer.to(self.device)

    def get_psnr(self, img1, img2, PIXEL_MAX=1.0):
        mse_ = torch.mean( (img1 - img2) ** 2 )
        return 10 * torch.log10(PIXEL_MAX / mse_)

    def bf_view(self, tensor):
        b,f,c,h,w = tensor.size()
        return tensor.reshape(b*f, c, h, w)

    def get_loss(self, sr, hr, ref, is_train, is_log, outs):
        # print('[get_loss]', sr.size(), hr.size(), ref.size())
        if is_log and 'vis' not in outs.keys():
            outs['vis'] = collections.OrderedDict()

        if len(list(sr.size())) == 5:
            sr = self.bf_view(sr)
            hr = self.bf_view(hr)
            ref = self.bf_view(ref)

        if hr.shape!=sr.shape:
            sr_down = F.interpolate(sr,scale_factor=1/self.scale,mode='bicubic',align_corners=False).clamp(0, 1)

        errs = collections.OrderedDict()
        errs['total'] = 0.

        if self.is_train:
            for i, l in enumerate(self.loss):
                loss = None

                if l['type'] == 'L1':
                    loss = l['function'](sr if not self.config.flag_HD_in else sr_down, hr)

                elif l['type'] == 'L1_lf':
                    sr_lf=self.gaussian_layer(sr if not self.config.flag_HD_in else sr_down)
                    hr_lf=self.gaussian_layer(hr)
                    loss = l['function'](sr_lf, hr_lf)

                elif ((l['type'] == 'FID_ref') or (l['type'] == 'FID_ref_X_mu')) and is_train:
                    loss, c = l['function'](sr, ref)
                    if is_log:
                        outs['vis']['contextual_ref_C'] = norm_res_vis(c)

                elif 'MFID_ref' in l['type'] and is_train:
                    b, c, h, w = sr.size()
                    b_ref, t, _, _, _ = ref.size()

                    sr_frame_batch = sr[:, None].expand(-1, t, -1, -1, -1).reshape(b*t, c, h, w)
                    ref_frame_batch = ref.reshape(b_ref*t, c, h, w)

                    loss, c = l['function'](sr_frame_batch,ref_frame_batch)
                    if is_log:
                        outs['vis']['contextual_ref_MFID_C'] = norm_res_vis(c)

                # elif l['type'] == 'FID_hr' and is_train:
                elif 'FID_hr' in l['type'] and is_train:
                    sr_ = sr if not self.config.flag_HD_in else sr_down
                    loss_sh, c_sh = l['function'](sr_, hr)
                    loss_hs, c_hs = l['function'](hr, sr_)
                    loss = loss_sh + loss_hs

                    if is_log:
                        outs['vis']['contextual_hr_C_sh'] = norm_res_vis(c_sh)
                        outs['vis']['contextual_hr_C_hs'] = norm_res_vis(c_hs)

                if loss is not None:
                    errs[l['type']] = l['weight'] * loss
                    errs['total'] += errs[l['type']]

        with torch.no_grad():
            errs['PSNR'] = self.get_psnr(sr if not self.config.flag_HD_in else sr_down, hr)
            # errs['LPIPS'] = torch.mean(self.LPIPS.forward(sr * 2. - 1., hr * 2. - 1.))

        return errs
