import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import json

import time
import numpy
import os
import sys
import collections
import numpy as np
import gc
import math
import random

from trainers import create_trainer
from utils import *
from ckpt_manager import CKPT_Manager
import warnings
warnings.filterwarnings("ignore")
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

class Runner():
    def __init__(self, config, rank = -1):
        self.rank = rank
        self.device = config.device
        if config.dist:
            self.pg = dist.new_group(range(dist.get_world_size()))

        self.config = config
        if self.rank <= 0:
            self.summary = SummaryWriter(config.LOG_DIR.log_scalar)

        ## model
        self.trainer = create_trainer(config)
        if self.rank <= 0 and config.is_verbose:
            self.trainer.print_network()

        ## checkpoint manager
        self.ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num, is_descending=True)

        ## training vars
        self.states = ['train', 'valid']
        #self.states = ['valid', 'train']
        self.max_epoch = int(math.ceil(config.total_itr / self.trainer.get_itr_per_epoch('train')))
        self.config.max_epoch = self.max_epoch

        if self.rank <= 0: print(toGreen('Max Epoch: {}'.format(self.max_epoch)))
        self.epoch_range = np.arange(1, self.max_epoch + 1)
        self.err_epoch = {'train': {}, 'valid': {}}
        self.norm = torch.tensor(0).to(self.device)
        self.lr = 0

        if (self.config.resume or self.config.resume_abs) is not None:
            if self.rank <= 0:
                remove_file_end_with(self.config.LOG_DIR.sample, '*.jpg')
                remove_file_end_with(self.config.LOG_DIR.sample, '*.png')
                remove_file_end_with(self.config.LOG_DIR.sample_val, '*.jpg')
                remove_file_end_with(self.config.LOG_DIR.sample_val, '*.png')
            if self.rank <= 0: print(toGreen('Resume Trianing...'))
            if self.rank <= 0: print(toRed('\tResuming {}..'.format(self.config.resume if self.config.resume is not None else self.config.resume_abs)))
            resume_state = self.ckpt_manager.resume(self.trainer.get_network(), self.config.resume, self.config.resume_abs, self.rank)
            if self.config.resume is not None:
                self.epoch_range = np.arange(resume_state['epoch'] + 1, self.max_epoch + 1)
                self.trainer.resume_training(resume_state)

    def train(self):
        # torch.backends.cudnn.benchmark = True
        if self.rank <= 0 : print(toYellow('\n\n=========== TRAINING START ============'))
        for epoch in self.epoch_range:
            ######
            if self.rank <= 0:
                if epoch % self.config.refresh_image_log_every_epoch['train'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.png')
                if epoch % self.config.refresh_image_log_every_epoch['valid'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.png')

            if self.rank <= 0 and epoch == 1:
                if self.config.resume is None:
                    self.ckpt_manager.save(self.trainer.get_network(), self.trainer.get_training_state(0), 0, score=[1e-8])
            # is_log = epoch == 1 or epoch % self.config.write_ckpt_every_epoch == 0 or epoch > self.max_epoch - 10
            is_log = epoch == 1 or epoch % self.config.write_ckpt_every_epoch == 0 or epoch > self.max_epoch - 1
            if self.config.resume is not None and epoch == int(self.config.resume) + 1:
                is_log = True
            ######

            for state in self.states:
                epoch_time = time.time()
                self.err_epoch[state] = {}
                self.norm = torch.tensor(0, dtype=torch.float, device='cuda')

                if state == 'train':
                    self.trainer.train()
                    self.iteration(epoch, state, is_log)
                elif state == 'valid' and is_log == True:
                    self.trainer.eval()
                    with torch.no_grad():
                        self.iteration(epoch, state, is_log)

                with torch.no_grad():
                    if is_log:
                        if config.dist:
                            dist.barrier()
                            dist.all_reduce(self.norm, op=dist.ReduceOp.SUM, group=self.pg, async_op=False)

                        for k, v in self.err_epoch[state].items():
                            if config.dist: dist.all_reduce(self.err_epoch[state][k], op=dist.ReduceOp.SUM, group=self.pg, async_op=False)
                            self.err_epoch[state][k] = (self.err_epoch[state][k] / self.norm).item()

                            if self.rank <= 0:
                                self.summary.add_scalar('{}_epoch/{}'.format(state, k), self.err_epoch[state][k], epoch)
                                self.summary.add_scalar('{}_itr/{}'.format(state, k), self.err_epoch[state][k], self.trainer.itr_global['train'])
                                # if state == 'train':
                                #     for name, param in self.trainer.get_network().named_parameters():
                                #         if any(check in name for check in ['weight']):
                                #             self.summary.add_histogram(name, param, self.trainer.itr_global['train'])
                                #         if param.grad is not None:
                                #             self.summary.add_histogram('grad.'+name, param.grad, self.trainer.itr_global['train'])

                        if self.rank <= 0:
                            if state == 'valid':
                                is_saved = False
                                while is_saved == False:
                                    #print(self.rank)
                                    try:
                                        self.ckpt_manager.save(self.trainer.get_network(), self.trainer.get_training_state(epoch), epoch, score=[self.err_epoch['valid']['PSNR'] if math.isnan(self.err_epoch['valid']['PSNR']) is False else -1 ])
                                        is_saved = True
                                    except Exception as ex:
                                        is_saved = False

                            if state == 'train':
                                print_logs(state.upper()+' TOTAL', self.config.mode, epoch, self.max_epoch, epoch_time, iter=self.trainer.itr_global[state], iter_total=self.config.total_itr, errs=self.err_epoch[state], log_etc=self.lr, is_overwrite=False)
                            else:
                                print_logs(state.upper()+' TOTAL', self.config.mode, epoch, self.max_epoch, epoch_time, note=config.note, errs=self.err_epoch[state], log_etc=self.lr, is_overwrite=False)
                                print('\n')
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def iteration(self, epoch, state, is_log):
        is_train = True if state == 'train' else False
        data_loader = self.trainer.data_loader_train if is_train else self.trainer.data_loader_eval
        if config.dist:
            if is_train: self.trainer.sampler_train.set_epoch(epoch)

        itr = 0
        itr_time = time.time()
        for inputs in data_loader:
            lr = None

            self.trainer.iteration(inputs, is_log, is_train)
            itr += 1

            with torch.no_grad():
                if is_log:
                    errs = self.trainer.results['errs']
                    norm = self.trainer.results['norm']
                    self.lr = self.trainer.results['log_etc']

                    for k, v in errs.items():
                        if itr == 1:
                            self.err_epoch[state][k] = v
                        else:
                            if k in self.err_epoch[state].keys():
                                self.err_epoch[state][k] += v
                            else:
                                self.err_epoch[state][k] = v
                    self.norm = self.norm + norm

                    if config.save_sample:
                        # saves image patches for logging
                        vis = self.trainer.results['vis']
                        sample_dir = self.config.LOG_DIR.sample if is_train else self.config.LOG_DIR.sample_val
                        # if itr == 1 or self.trainer.itr_global[state] % config.write_log_every_itr[state] == 0:
                        if (state == 'train' and (itr * self.trainer.itr_inc[state]) % config.write_log_every_itr[state] == 0) or \
                                (state == 'valid' and (itr * self.trainer.itr_inc[state]) % config.write_log_every_itr[state] == 0):

                            try:
                                i = 1
                                for key, val in vis.items():
                                    if val.dim() == 5:
                                        for j in range(val.size()[1]):
                                            vutils.save_image(val[:, j, :, :, :], '{}/E{:02}_R{:02}_I{:06}_{:02}_{}_{:03}.{}'.format(sample_dir, epoch, self.rank, self.trainer.itr_global[state], i, key, j, 'png' if 'png' in key else 'jpg'), nrow=math.ceil(math.sqrt(val.size()[0])), padding = 0, normalize = False)
                                    else:
                                        vutils.save_image(val, '{}/E{:02}_R{:02}_I{:06}_{:02}_{}.{}'.format(sample_dir, epoch, self.rank, self.trainer.itr_global[state], i, key, 'png' if 'png' in key else 'jpg'), nrow=math.ceil(math.sqrt(val.size()[0])), padding = 0, normalize = False)
                                    i += 1
                            except Exception as ex:
                                print_err(key)
                                print_err(ex)

                    if self.rank <= 0:
                        errs_itr = collections.OrderedDict()
                        for k, v in errs.items():
                            errs_itr[k] = v / norm
                        ## if you are using DDP, itr and total itr may not exaclty match, which is because GPU0 (rank0) may be handling shorter video clips
                        print_logs(state.upper(), self.config.mode, epoch, self.max_epoch, itr_time, itr * self.trainer.itr_inc[state], self.trainer.get_itr_per_epoch(state), errs = errs_itr, log_etc = self.lr, is_overwrite = itr > 1)
                        # print('\n')
                        itr_time = time.time()

##########################################################
def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

if __name__ == '__main__':
    project = 'RefVSR_CVPR2022'
    mode = 'RefVSR'

    from configs.config import set_data_path
    import importlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', action = 'store_true', default = False, help = 'whether to delete log')
    parser.add_argument('--config', type = str, default = None, help = 'config name') # do not change the default value
    parser.add_argument('--mode', type = str, default = mode, help = 'mode name')
    parser.add_argument('--project', type = str, default = project, help = 'project name')
    parser.add_argument('-data', '--data', type=str, default = 'VRefSR', help = 'dataset to train or test (VRefSR|CUFED5)')
    parser.add_argument('-LRS', '--LRS', type=str, default = 'CA', help = 'learning rate scheduler to use [LD or CA]')
    parser.add_argument('-b', '--batch_size', type = int, default = 8, help = 'number of batch')
    args, _ = parser.parse_known_args()

    if args.is_train:
        config_lib = importlib.import_module('configs.{}'.format(args.config))
        config = config_lib.get_config(args.project, args.mode, args.config, args.data, args.LRS, args.batch_size)
        config.is_train = True

        ## DEFAULT
        parser.add_argument('-trainer', '--trainer', type = str, default = 'trainer', help = 'model name')
        parser.add_argument('-net', '--network', type = str, default = 'MCSR', help = 'network name')
        parser.add_argument('-loss', '--loss', type = str, default = config.loss, help = 'loss')
        parser.add_argument('-r', '--resume', type = str, default = config.resume, help = 'name of state or ckpt (names are the same)')
        parser.add_argument('-ra', '--resume_abs', type = str, default = config.resume_abs, help = 'absolute path of state or ckpt')
        parser.add_argument('-dl', '--delete_log', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument('-lr', '--lr_init', type = float, default = config.lr_init, help = 'leraning rate')
        parser.add_argument('-th', '--thread_num', type = int, default = config.thread_num, help = 'number of thread')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = config.dist, help = 'whether to distributed pytorch')
        parser.add_argument('-vs', '--is_verbose', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument('-ss', '--save_sample', action = 'store_true', default = False, help = 'whether to save_sample')
        parser.add_argument('-is_crop_valid', '--is_crop_valid', action = 'store_true', default = False, help = 'whether to check train-val memory')
        parser.add_argument('-note', '--note', type = str, default = config.note, help = 'note')
        parser.add_argument("--local_rank", type=int)

        ## CUSTOM
        parser.add_argument('-wi', '--weights_init', type = float, default = config.wi, help = 'weights_init')
        parser.add_argument('-win', '--weights_init_normal', type = float, default = config.win, help = 'weights_init')
        parser.add_argument('-proc', '--proc', type = str, default = 'proc', help = 'dummy process name for killing')
        parser.add_argument('-gc', '--gc', type = float, default = config.gc, help = 'gradient clipping')

        parser.add_argument('-frame_num', '--frame_num', type=int, default = config.frame_num)

        args, _ = parser.parse_known_args()

        ## default
        config.trainer = args.trainer
        config.network = args.network
        config.loss = args.loss

        config.resume = args.resume
        config.resume_abs = args.resume_abs
        config.delete_log = False if (config.resume or config.resume_abs) is not None else args.delete_log
        config.lr_init = args.lr_init
        config.batch_size = args.batch_size
        config.thread_num = args.thread_num
        config.dist = args.dist
        config.data = args.data
        config.LRS = args.LRS
        config.is_verbose = args.is_verbose
        config.save_sample = args.save_sample
        config.note = args.note
        config.is_crop_valid = args.is_crop_valid

        # CUSTOM
        config.wi = args.weights_init
        config.win = args.weights_init_normal
        config.gc = args.gc

        config.frame_num = args.frame_num

        # set datapath
        config = set_data_path(config, config.data, is_train=True)

        if config.dist:
            init_dist()
            rank = dist.get_rank()
        else:
            rank = -1

        if rank <= 0:
            handle_directory(config, config.delete_log)
            print(toGreen('Laoding Config...'))
            config_lib.print_config(config)
            config_lib.log_config(config.LOG_DIR.config, config)
            print(toRed('\tProject : {}'.format(config.project)))
            print(toRed('\tMode : {}'.format(config.mode)))
            print(toRed('\tConfig: {}'.format(config.config)))
            print(toRed('\tNetwork: {}'.format(config.network)))
            print(toRed('\tTrainer: {}'.format(config.trainer)))
            print(toRed('\tLR scheduler: {}'.format(config.LRS)))

        if config.dist:
            dist.barrier()

        ## random seed
        seed = config.manual_seed
        if seed is None:
            seed = random.randint(1, 10000)
        if rank <= 0 and config.is_verbose: print('Random seed: {}'.format(seed))

        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        runner = Runner(config, rank)
        if config.dist:
            dist.barrier()
        runner.train()

    else:
        from eval import *
        from configs.config import get_config, set_data_path
        from easydict import EasyDict as edict
        print(toGreen('Laoding Config for evaluation'))
        if args.config is None:
            config = get_config(args.project, args.mode, None)
            with open('{}/config.txt'.format(config.LOG_DIR.config)) as json_file:
                json_data = json.load(json_file)
                # config_lib = importlib.import_module('configs.{}'.format(json_data['config']))
                config = edict(json_data)
                # print(config['config'])
        else:
            config_lib = importlib.import_module('configs.{}'.format(args.config))
            config = config_lib.get_config(args.project, args.mode, args.config)

        config.is_train = False
        ## EVAL
        parser.add_argument('-net', '--network', type = str, default = config.network, help = 'network name')
        parser.add_argument('-ckpt_name', '--ckpt_name', type=str, default = None, help='ckpt name')
        parser.add_argument('-ckpt_abs_name', '--ckpt_abs_name', type=str, default = None, help='ckpt abs name')
        parser.add_argument('-ckpt_epoch', '--ckpt_epoch', type=int, default = None, help='ckpt epoch')
        parser.add_argument('-ckpt_sc', '--ckpt_score', action = 'store_true', help='ckpt name')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = False, help = 'whether to distributed pytorch')
        parser.add_argument('-eval_mode', '--eval_mode', type=str, default = 'qual_quan', help = 'evaluation mode. qual(qualitative)/quan(quantitative)')
        parser.add_argument('-test_set', '--test_set', type=str, default = 'test', help = 'test set to evaluate. test/valid')
        parser.add_argument('-is_qual', '--is_qual', action = 'store_true', default = False, help = 'whether to save image')
        parser.add_argument('-is_debug', '--is_debug', action = 'store_true', default = False, help = 'whether to be in debug mode')
        parser.add_argument('-frame_num', '--frame_num', type=int, default = config.frame_num)
        parser.add_argument('-vid_name', '--vid_name', nargs='+', default = None, help = 'Name of video(s) to evaluate. e.g., --vid_name 0024 0074 ')
        parser.add_argument('-ss', '--save_sample', action = 'store_true', default = False, help = 'whether to save_sample')
        args, _ = parser.parse_known_args()

        config.network = args.network
        config.frame_num = args.frame_num
        config.center_idx = config.frame_num//2
        config.EVAL.ckpt_name = args.ckpt_name
        config.EVAL.ckpt_abs_name = args.ckpt_abs_name
        config.EVAL.ckpt_epoch = args.ckpt_epoch
        config.EVAL.is_qual = args.is_qual
        config.EVAL.is_debug = args.is_debug
        config.EVAL.load_ckpt_by_score = args.ckpt_score
        config.EVAL.vid_name = args.vid_name
        config.save_sample = args.save_sample

        config.dist = args.dist
        config.EVAL.eval_mode = args.eval_mode
        config.EVAL.test_set = args.test_set
        config.EVAL.data = args.data
        config = set_data_path(config, config.EVAL.data, is_train=False)

        print(toRed('\tProject : {}'.format(config.project)))
        print(toRed('\tMode : {}'.format(config.mode)))
        print(toRed('\tConfig: {}'.format(config.config)))
        print(toRed('\tNetwork: {}'.format(config.network)))
        print(toRed('\tTrainer: {}'.format(config.trainer)))

        eval(config)
