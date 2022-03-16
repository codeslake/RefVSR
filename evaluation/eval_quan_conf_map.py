import torch
import torchvision.utils as vutils

import os
import sys
import datetime
import gc
import numpy as np
import cv2
import collections

from .init import init
from .metrics import psnr, ssim
from utils import *
from data_loader.utils import refine_image_pt, read_frame, load_file_list, norm


def eval_quan_conf_map(config):
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap('inferno')

    mode = config.EVAL.eval_mode
    config.save_sample = True
    network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name = init(config, mode)

    ##
    total_norm = 0
    frame_len_prev = 0
    total_itr_time = 0
    total_itr_time_video = 0

    for i, inputs in enumerate(model.data_loader_eval):
        is_first_frame = inputs['is_first'][0].item()

        # for k in inputs.keys():
        #     print(k, inputs[k].size())
        if 'is_continue' in inputs.keys() and inputs['is_continue'][0].item():
            print('passing, video', inputs['video_name'][0])
            frame_len_prev += 1
            continue

        if is_first_frame:
            if i > 0:
                total_itr_time = total_itr_time + total_itr_time_video
                total_itr_time_video = total_itr_time_video / frame_len_prev

                print('[MEAN EVAL {}|{}|{}][{}/{}] ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0], inputs['video_len'][0], total_itr_time_video))
                with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
                    file.write('[MEAN EVAL {}|{}|{}][{}/{}] ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0], inputs['video_len'][0], total_itr_time_video))
                    file.close()

            total_itr_time_video = 0

        #########################
        init_time = time.time()
        results = model.evaluation(inputs, is_log=True)
        itr_time = time.time() - init_time
        #########################

        ## evaluation
        errs = results['errs']
        outs = results['vis']
        vis = results['vis']['eval_vis']
        conf_map_norm = vis['conf_map']

        if 'conf_map_prop' in vis.keys():
            conf_map_prop_norm = vis['conf_map_prop']
            conf_map_prop_norm = conf_map_prop_norm - conf_map_prop_norm.min()
            conf_map_prop_norm = conf_map_prop_norm / conf_map_prop_norm.max()

            conf_map_prop_b_norm = vis['conf_map_prop_backward']
            conf_map_prop_b_norm = conf_map_prop_b_norm - conf_map_prop_b_norm.min()
            conf_map_prop_b_norm = conf_map_prop_b_norm / conf_map_prop_b_norm.max()

            conf_map_prop_f_norm = vis['conf_map_prop_forward']
            conf_map_prop_f_norm = conf_map_prop_f_norm - conf_map_prop_f_norm.min()
            conf_map_prop_f_norm = conf_map_prop_f_norm / conf_map_prop_f_norm.max()

            conf_map_norm = conf_map_norm - conf_map_norm.min()
            conf_map_norm = conf_map_norm / conf_map_norm.max()

            conf_map_norm_cpu = conf_map_norm.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
            conf_map_norm_cpu = colormap(conf_map_norm_cpu)[:, :, :3]
            conf_map_norm = torch.Tensor(conf_map_norm_cpu)[None, :].permute(0, 3, 1, 2)

            conf_map_prop_norm_cpu = conf_map_prop_norm.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
            conf_map_prop_norm_cpu = colormap(conf_map_prop_norm_cpu)[:, :, :3]
            conf_map_prop_norm = torch.Tensor(conf_map_prop_norm_cpu)[None, :].permute(0, 3, 1, 2)

            conf_map_prop_b_norm_cpu = conf_map_prop_b_norm.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
            conf_map_prop_b_norm_cpu = colormap(conf_map_prop_b_norm_cpu)[:, :, :3]
            conf_map_prop_b_norm = torch.Tensor(conf_map_prop_b_norm_cpu)[None, :].permute(0, 3, 1, 2)

            conf_map_prop_f_norm_cpu = conf_map_prop_f_norm.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
            conf_map_prop_f_norm_cpu = colormap(conf_map_prop_f_norm_cpu)[:, :, :3]
            conf_map_prop_f_norm = torch.Tensor(conf_map_prop_f_norm_cpu)[None, :].permute(0, 3, 1, 2)
        else:
            conf_map_norm_cpu = conf_map_norm.cpu().numpy()[0].transpose(1, 2, 0)[:, :, 0]
            conf_map_norm_cpu = colormap(conf_map_norm_cpu)[:, :, :3]
            conf_map_norm = torch.Tensor(conf_map_norm_cpu)[None, :].permute(0, 3, 1, 2)

            conf_map_prop_norm = None
            conf_map_prop_b_norm = None
            conf_map_prop_f_norm = None


        try:
            inp = outs['LR_UW_png']
            output = outs['SR_UW_png']
        except:
            inp = outs['LR_UW']
            output = outs['SR_UW']

        frame_name = inputs['frame_name'][0]
        print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, itr_time))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'w' if (i == 0) else 'a') as file:
            file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, itr_time))
            file.close()

        # qualitative
        inp_cpu = inp.cpu().numpy()[0].transpose(1, 2, 0)
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
        if 'gt' in inputs.keys():
            gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)

        conf_map_norm_cpu = conf_map_norm.cpu().numpy()[0].transpose(1, 2, 0)
        if conf_map_prop_norm is not None:
            conf_map_prop_norm_cpu = conf_map_prop_norm.cpu().numpy()[0].transpose(1, 2, 0)
        if conf_map_prop_b_norm is not None:
            conf_map_prop_b_norm_cpu = conf_map_prop_b_norm.cpu().numpy()[0].transpose(1, 2, 0)
        if conf_map_prop_f_norm is not None:
            conf_map_prop_f_norm_cpu = conf_map_prop_f_norm.cpu().numpy()[0].transpose(1, 2, 0)

        for iformat in ['png', 'jpg']:
        # for iformat in ['jpg']:
            frame_name_no_ext = frame_name.split('.')[0]
            save_path_deblur = os.path.join(save_path_root_deblur, iformat)
            Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

            Path(os.path.join(save_path_deblur, 'input', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
            save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
            cv2.imwrite(save_file_path_deblur_input, cv2.cvtColor(inp_cpu*255, cv2.COLOR_RGB2BGR))

            Path(os.path.join(save_path_deblur, 'output', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
            save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
            cv2.imwrite(save_file_path_deblur_output, cv2.cvtColor(output_cpu*255, cv2.COLOR_RGB2BGR))

            Path(os.path.join(save_path_deblur, 'conf_map_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
            save_file_path_deblur_conf_map_norm = os.path.join(save_path_deblur, 'conf_map_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
            cv2.imwrite(save_file_path_deblur_conf_map_norm, cv2.cvtColor(conf_map_norm_cpu*255, cv2.COLOR_RGB2BGR))

            if conf_map_prop_norm is not None:
                Path(os.path.join(save_path_deblur, 'conf_map_prop_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_conf_map_prop_norm = os.path.join(save_path_deblur, 'conf_map_prop_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                cv2.imwrite(save_file_path_deblur_conf_map_prop_norm, cv2.cvtColor(conf_map_prop_norm_cpu*255, cv2.COLOR_RGB2BGR))

            if conf_map_prop_b_norm is not None:
                Path(os.path.join(save_path_deblur, 'conf_map_prop_b_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_conf_map_prop_b_norm = os.path.join(save_path_deblur, 'conf_map_prop_b_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                cv2.imwrite(save_file_path_deblur_conf_map_prop_b_norm, cv2.cvtColor(conf_map_prop_b_norm_cpu*255, cv2.COLOR_RGB2BGR))

            if conf_map_prop_f_norm is not None:
                Path(os.path.join(save_path_deblur, 'conf_map_prop_f_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_conf_map_prop_f_norm = os.path.join(save_path_deblur, 'conf_map_prop_f_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                cv2.imwrite(save_file_path_deblur_conf_map_prop_f_norm, cv2.cvtColor(conf_map_prop_f_norm_cpu*255, cv2.COLOR_RGB2BGR))

            if 'gt' in inputs.keys():
                Path(os.path.join(save_path_deblur, 'gt', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_gt = os.path.join(save_path_deblur, 'gt', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(gt, '{}'.format(save_file_path_deblur_gt), nrow=1, padding = 0, normalize = False)

        total_itr_time_video = total_itr_time_video + itr_time
        total_norm = total_norm + 1
        frame_len_prev = inputs['frame_len'][0]

    # total average
    total_itr_time = (total_itr_time + total_itr_time_video) / total_norm

    sys.stdout.write('\n[TOTAL {}|{}] ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, total_itr_time))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
        file.write('\n[TOTAL {}|{}] ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, total_itr_time))
        file.close()