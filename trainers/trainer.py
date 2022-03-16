import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import collections
from shutil import copy2
import gc

import numpy as np

from utils import toGreen, toRed, print_err
from data_loader.utils import refine_image_pt
from trainers.utils import clone_detach_dict
from trainers.baseTrainer import baseTrainer

from models.SRNet import SRNet
from models.loss.Loss import Loss
from models.utils import warp

import fvcore.nn as fvnn
from ptflops import get_model_complexity_info

class Trainer(baseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)

        self.device = config.device
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.ws = torch.distributed.get_world_size() if config.dist else 1
        self.itr_global = {'train': 0, 'valid': 0}
        self.itr_inc = {'train': self.config.frame_itr_num, 'valid': 1}

        ### NETWORKS ###
        if self.rank <= 0 : print(toGreen('Loading Model...'))
        self.network = SRNet(config).to(self.device, non_blocking=True)


        if self.is_train and self.config.resume is None or self.is_train and os.path.exists('./models/archs/{}.py'.format(config.network)):
            copy2('./models/archs/{}.py'.format(config.network), self.config.LOG_DIR.offset)
            copy2('./trainers/{}.py'.format(config.trainer), self.config.LOG_DIR.offset)

        ### INIT for training ###
        if self.is_train:
            if self.config.is_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            self.network.init()
            self._set_dataloader()
            self._set_optim()
            self._set_lr_scheduler()
            self.Loss = Loss(config)
            if config.is_verbose:
                for name, param in self.network.named_parameters():
                    if self.rank <= 0: print(name, ', ', param.requires_grad)
        else:
            self._set_dataloader(self.is_train)

        ### DDP ###
        if config.cuda:
            if config.dist:
                if self.rank <= 0: print(toGreen('Building Dist Parallel Model...'))
                self.network = DDP(self.network, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), broadcast_buffers=True, find_unused_parameters=False)
            else:
                self.network = DP(self.network).to(self.device, non_blocking=True)

            # if self.config.is_train and self.rank <= 0:
            ### PROFILE ###
            if self.rank <= 0:
                #with torch.no_grad():
                #     # input_size = (1, self.config.frame_num, 3, 1080//4, 1920//4)
                #     input_size = (1, self.config.frame_num, 3, 256, 256)
                #     if config.dist:
                #         inputs = self.network.module.input_constructor(input_size)
                #     else:
                #         try:
                #             inputs = self.network.input_constructor(input_size)
                #         except:
                #             inputs = self.network.module.input_constructor(input_size)
                #     inputs = tuple(inputs.values())
                #     Macs = fvnn.FlopCountAnalysis(self.network, inputs=inputs).total()
                #     params = fvnn.parameter_count(self.network)['']
                with torch.no_grad():
                    if config.flag_HD_in is False:
                        res = (1, config.frame_num, 3, 1080//config.scale, 1090//config.scale)
                    else:
                        res = (1, config.frame_num, 3, 1080, 1920)
                    Macs,params = get_model_complexity_info(self.network, res, input_constructor = self.network.module.input_constructor, as_strings=False, print_per_layer_stat=config.is_verbose)

            if self.rank <= 0:
                print(toGreen('Computing model complexity...'))
                print('{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
                print('{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2))
                if self.is_train:
                    with open(config.LOG_DIR.offset + '/cost.txt', 'w') as f:
                        f.write('{:<30}  {:<8} B\n'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
                        f.write('{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2))
                        f.close()

            # gc.collect()
            # torch.cuda.empty_cache()

    ######################################################################################################
    ########################### Edit from here for training/testing scheme ###############################
    ######################################################################################################

    def _set_results(self, inputs, outs, errs, log, norm_=1, is_train=False):
        ## save visuals (inputs)
        self.results['vis'] = collections.OrderedDict()
        self.results['vis']['LR_UW{}'.format('' if not self.config.flag_HD_in else '')] = outs['LR_UW']
        self.results['vis']['SR_UW{}'.format('' if not self.config.flag_HD_in else '')] = outs['result'].detach()
        self.results['vis']['HR_UW'] = outs['HR_UW']
        self.results['vis']['LR_REF_W'] = outs['LR_REF_W']
        if is_train:
            self.results['vis']['HR_REF_W'] = outs['HR_REF_W']

        if self.config.save_sample:
            ## save visuals (outputs)
            if 'vis' in outs.keys():
                self.results['vis'].update(outs['vis'].items())

        # rotate roated portrait frames (which have been rotated 90 in data loader) to original rotation (due to memory issue during valudation in training)
        # if 'is_portrait' in inputs.keys() and inputs['is_portrait']:
        #     for k, v in self.results['vis'].items():
        #         self.results['vis'][k] = torch.rot90(v, 3, [v.dim()-2, v.dim()-1])
        # print('\n\n', self.rank, inputs['is_portrait'], inputs['LR_UW'].size(), inputs['LR_UW'].is_contiguous(), self.results['vis']['SR_UW'].size(), '\n\n')

        ## Essentials ##
        # save scalars
        self.results['errs'] = errs
        self.results['norm'] = norm_
        # learning rate
        self.results['log_etc'] = log

    def iteration(self, inputs, is_log, is_train):
        ## Init for logging
        state = 'train' if is_train else 'valid'
        self.itr_global[state] += self.itr_inc[state]

        ## Prepare data
        LR_UW_total_frames = refine_image_pt(inputs['LR_UW'].to(self.device, non_blocking=True), self.config.refine_val_lr)
        LR_REF_W_total_frames = refine_image_pt(inputs['LR_REF_W'].to(self.device, non_blocking=True), self.config.refine_val_lr)
        HR_UW_total_frames = refine_image_pt(inputs['HR_UW'].to(self.device, non_blocking=True), self.config.refine_val_hr)

        if self.config.is_use_T:
            LR_REF_T_total_frames = refine_image_pt(inputs['LR_REF_T'].to(self.device, non_blocking=True), self.config.refine_val_lr) if self.config.is_use_T else None
            HR_REF_T_total_frames = refine_image_pt(inputs['HR_REF_T'].to(self.device, non_blocking=True), self.config.refine_val_hr) if self.config.is_use_T else None

        HR_REF_W_total_frames = refine_image_pt(inputs['HR_REF_W'].to(self.device, non_blocking=True), self.config.refine_val_lr)

        norm_ = 0
        b, total_frame_num, c, h, w = LR_UW_total_frames.size()
        errs_total = collections.OrderedDict()

        ## Iteration
        for i in range(total_frame_num-(self.config.frame_num - 1)):
            is_first_frame = i == 0 if is_train else inputs['is_first'][0].item()

            LR_UW_frames = LR_UW_total_frames[:, i:i+self.config.frame_num]
            LR_REF_W_frames = LR_REF_W_total_frames[:, i:i+self.config.frame_num]
            HR_UW_frames = HR_UW_total_frames[:, i:i+self.config.frame_num]
            HR_UW = HR_UW_frames[:, self.config.frame_num//2]

            #################################################################################################

            if self.config.is_amp:
                with torch.cuda.amp.autocast():
                    outs = self.network(LR_UW_frames, LR_REF_W_frames, is_first_frame, is_log, is_train)
            else:
                outs = self.network(LR_UW_frames, LR_REF_W_frames, is_first_frame, is_log, is_train)

            #################################################################################################

            if self.config.is_use_T:
                REF_frames = HR_REF_T_total_frames[:, :i+self.config.frame_num]
            else:
                REF_frames = HR_REF_W_total_frames[:, :i+self.config.frame_num]

            errs = self.Loss.get_loss(outs['result'], HR_UW, REF_frames, is_train, is_log, outs)


            ## Updating network & get log (learning rate, gnorm)
            if self.config.is_amp:
                log = self._update_amp(errs) if is_train else None
            else:
                log = self._update(errs) if is_train else None

            ## Loggging
            with torch.no_grad():
                norm_ += b
                if errs is not None:
                    for k, v in errs.items():
                        v_t = 0 if i == 0 else errs_total[k]
                        # errs_total[k] = v_t + v.item() if isinstance(v, torch.Tensor) else v * b
                        errs_total[k] = v_t + b * v.detach().clone() if isinstance(v, torch.Tensor) else v

        assert norm_ != 0
        ## setting results for the log
        if is_log:
            outs['LR_UW'] = LR_UW_frames[:, self.config.frame_num//2]
            outs['HR_UW'] = HR_UW_frames[:, self.config.frame_num//2]
            outs['LR_REF_W'] = LR_REF_W_frames[:, self.config.frame_num//2]
            if is_train:
                HR_REF_W_frames = HR_REF_W_total_frames[:, i:i+self.config.frame_num]
                outs['HR_REF_W'] = HR_REF_W_frames[:, self.config.frame_num//2]
            if self.config.is_use_T and 'vis' in outs.keys():
                LR_REF_T_frames = LR_REF_T_total_frames[:, i:i+self.config.frame_num]
                outs['vis']['LR_REF_T'] = LR_REF_T_frames[:, self.config.frame_num//2]
                if is_train:
                    HR_REF_T_frames = HR_REF_T_total_frames[:, i:i+self.config.frame_num]
                    outs['vis']['HR_REF_T'] = HR_REF_T_frames[:, self.config.frame_num//2]

            if 'vis' not in outs.keys():
                outs['vis'] = collections.OrderedDict()
            else:
                pass

            self._set_results(inputs, clone_detach_dict(outs), errs_total, log, norm_, is_train)

    def evaluation(self, inputs, is_log=False, is_PSNR=False):
        # prepare data
        LR_UW_total_frames = refine_image_pt(inputs['LR_UW'].to(self.device, non_blocking=True), self.config.refine_val_lr).contiguous()
        LR_REF_W_total_frames = refine_image_pt(inputs['LR_REF_W'].to(self.device, non_blocking=True), self.config.refine_val_lr).contiguous()

        b, total_frame_num, c, h, w = LR_UW_total_frames.size()
        errs_total = collections.OrderedDict()

        is_first_frame = inputs['is_first'][0].item()

        LR_UW_frames = LR_UW_total_frames[:, :self.config.frame_num]
        LR_REF_W_frames = LR_REF_W_total_frames[:, :self.config.frame_num]

        #################################################################################################

        if self.config.is_amp:
            with torch.cuda.amp.autocast():
                outs = self.network(LR_UW_frames, LR_REF_W_frames, is_first_frame, is_log=is_log, is_train=False)
        else:
            outs = self.network(LR_UW_frames, LR_REF_W_frames, is_first_frame, is_log=is_log, is_train=False)

        #################################################################################################

        ## setting results for the log
        result = collections.OrderedDict()
        result['vis'] = collections.OrderedDict()
        result['vis']['LR_UW'] = LR_UW_frames[:, self.config.frame_num//2]
        result['vis']['SR_UW'] = outs['result']
        result['vis']['HR_UW'] = refine_image_pt(inputs['HR_UW'][:, self.config.frame_num//2].to(self.device, non_blocking=True), self.config.refine_val_hr) 
        result['vis']['eval_vis'] = outs['eval_vis'] if 'eval_vis' in outs.keys() else None
        if is_PSNR:
            mse_ = torch.mean( (result['vis']['SR_UW'] - result['vis']['HR_UW']) ** 2 )
            PSNR = 10 * torch.log10(1 / mse_)
            errs = collections.OrderedDict()
            errs['PSNR'] = PSNR
        else:
            errs = None
        result['errs'] = errs
        return result
