from easydict import EasyDict as edict
import json
import os
import collections
import numpy as np
import torch

def get_config(project = '', mode = '', config_ = '', data = '', LRS = '', batch_size = 8):
    ## GLOBAL
    config = edict()

    config.project = project
    config.mode = mode
    config.config = config_
    config.is_train = False
    config.thread_num = batch_size
    config.cuda = True
    config.dist = False
    config.resume = None # 'resume epoch'
    config.resume_abs = None # 'resume abs name'
    config.manual_seed = 0
    config.is_verbose = False
    config.save_sample = False
    config.loss = None
    config.note = None # note for log
    config.is_crop_valid = False
    config.crop_valid_offset = 12 # if config.is_crop_valid==Ture, LR and Ref images whill be cropped e.g., LR[crop_offset:-crop_offset, :, :]


    ##################################### TRAIN #####################################
    if config.cuda == True:
        config.device = 'cuda'
    else:
        config.device = 'cpu'
    config.trainer = ''
    config.network = ''

    config.batch_size = batch_size
    config.batch_size_test = 1
    config.patch_size = 64
    config.actual_patch_size = None #(crop during iteration -mem issue)

    # learning rate
    config.lr_init = 1e-4
    config.gc = 1.0 # gradient clipping

    ## Naive Decay
    config.LRS = LRS # LD or CA

    # adam
    config.beta1 = 0.9

    # data dir
    config.data = 'RealMCVSR'
    config.data_offset = '/data1/junyonglee'
    config.HR_data_path = None
    config.LR_data_path = None
    config.is_use_T = False ## temporal
    config.is_crop = False ## temporal

    # logs
    config.max_ckpt_num = 100
    config.write_ckpt_every_epoch = 4
    config.refresh_image_log_every_epoch = {'train':config.write_ckpt_every_epoch*4, 'valid':config.write_ckpt_every_epoch*4}
    config.write_log_every_itr = {'train':26, 'valid': 10}

    # log dirs
    config.LOG_DIR = edict()
    log_offset = '/Jarvis/logs/junyonglee'
    offset = os.path.join(log_offset, config.project)
    offset = os.path.join(offset, '{}'.format(mode))
    config.LOG_DIR.offset = offset
    config.LOG_DIR.ckpt = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch')
    config.LOG_DIR.ckpt_ckpt = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch', 'ckpt')
    config.LOG_DIR.ckpt_state = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch', 'state')
    config.LOG_DIR.log_scalar = os.path.join(config.LOG_DIR.offset, 'log', 'train', 'scalar')
    config.LOG_DIR.log_image = os.path.join(config.LOG_DIR.offset, 'log', 'train', 'image', 'train')
    config.LOG_DIR.sample = os.path.join(config.LOG_DIR.offset, 'sample', 'train')
    config.LOG_DIR.sample_val = os.path.join(config.LOG_DIR.offset, 'sample', 'valid')
    config.LOG_DIR.config = os.path.join(config.LOG_DIR.offset, 'config')

    ################################## VALIDATION ###################################
    # data path
    config.VAL = edict()
    config.VAL.HR_data_path = None
    config.VAL.LR_data_path = None

    ##################################### EVAL ######################################
    config.EVAL = edict()
    config.EVAL.eval_mode = 'qual_quan' # qual
    config.EVAL.is_qual = False
    config.EVAL.is_debug = True
    
    config.EVAL.data = 'RealMCVSR'
    config.EVAL.test_set = 'test'

    config.EVAL.load_ckpt_by_score = True
    config.EVAL.ckpt_name = None
    config.EVAL.ckpt_epoch = None
    config.EVAL.ckpt_abs_name = None
    config.EVAL.low_res = False
    config.EVAL.ckpt_load_path = None

    # data dir
    config.EVAL.HR_data_path = None
    config.EVAL.LR_data_path = None

    # log dir
    config.EVAL.LOG_DIR = edict()
    config.EVAL.LOG_DIR.save = os.path.join(config.LOG_DIR.offset, 'result')

    return config

def set_data_path(config, data, is_train):
    if data == 'RealMCVSR':
        if config.flag_HD_in is False:
            lr_path = 'LRx2' if config.scale == 2 else 'LRx4'
            hr_ref_W_path = 'LRx2'
            hr_ref_T_path = 'LRx4'
        else:
            lr_path = 'HR'
            hr_ref_W_path = 'HR'
            hr_ref_T_path = 'HR'

        if is_train:
            config.LR_data_path = os.path.join(config.data_offset, data, 'train', lr_path)
            config.HR_data_path = os.path.join(config.data_offset, data, 'train', 'HR')
            config.HR_ref_data_W_path = os.path.join(config.data_offset, data, 'train', hr_ref_W_path)
            config.HR_ref_data_T_path = os.path.join(config.data_offset, data, 'train', hr_ref_T_path)

            config.VAL.LR_data_path = os.path.join(config.data_offset, data, 'valid', lr_path)
            config.VAL.HR_data_path = os.path.join(config.data_offset, data, 'valid', 'HR')
            config.VAL.HR_ref_data_W_path = os.path.join(config.data_offset, data, 'valid', hr_ref_W_path)
            config.VAL.HR_ref_data_T_path = os.path.join(config.data_offset, data, 'valid', hr_ref_T_path)
        else:
            config.EVAL.LR_data_path = os.path.join(config.data_offset, data, config.EVAL.test_set, lr_path)
            config.EVAL.HR_data_path = os.path.join(config.data_offset, data, config.EVAL.test_set, 'HR')
            config.EVAL.HR_ref_data_W_path = os.path.join(config.data_offset, data, config.EVAL.test_set, hr_ref_W_path)
            config.EVAL.HR_ref_data_T_path = os.path.join(config.data_offset, data, config.EVAL.test_set, hr_ref_T_path)
            config.EVAL.vid_name = None
            # config.EVAL.vid_name = ['0024', '0074', '0121']

        config.UW_path = 'UW'
        config.W_path = 'W'
        config.T_path = 'T'

    return config

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write(json.dumps(cfg, indent=4))
        f.close()

def print_config(cfg):
    print(json.dumps(cfg, indent=4))

