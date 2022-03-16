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

def eval_qual_quan(config):
    mode = config.EVAL.eval_mode
    network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name = init(config, mode)

    ##
    total_norm = 0
    total_itr_time = PSNR_mean_total = SSIM_mean_total = 0
    total_itr_time_video = PSNR_mean = SSIM_mean = 0
    frame_len_prev = 0

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
                PSNR_mean_total = PSNR_mean_total + PSNR_mean
                SSIM_mean_total = SSIM_mean_total + SSIM_mean
                total_itr_time = total_itr_time + total_itr_time_video

                PSNR_mean = PSNR_mean / frame_len_prev
                SSIM_mean = SSIM_mean / frame_len_prev
                total_itr_time_video = total_itr_time_video / frame_len_prev

                print('[MEAN EVAL {}|{}|{}][{}/{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0], inputs['video_len'][0], PSNR_mean, SSIM_mean, total_itr_time_video))
                with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
                    file.write('[MEAN EVAL {}|{}|{}][{}/{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0], inputs['video_len'][0], PSNR_mean, SSIM_mean, total_itr_time_video))
                    file.close()

            total_itr_time_video = PSNR_mean = SSIM_mean = 0

        #########################
        init_time = time.time()
        with torch.no_grad():
            results = model.evaluation(inputs, is_PSNR=not config.EVAL.qualitative_only)
        gc.collect()
        torch.cuda.empty_cache()
        itr_time = time.time() - init_time
        #########################

        ## evaluation
        errs = results['errs']
        outs = results['vis']

        try:
            inp = outs['LR_UW_png']
            output = outs['SR_UW_png']
        except:
            inp = outs['LR_UW']
            output = outs['SR_UW']

        PSNR = SSIM = 0
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
        gt = outs['HR_UW']
        gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)

        if not config.EVAL.qualitative_only:
            if 'SR_UW_png' in outs.keys() or 'SR_UW' in outs.keys():

                # PSNR = psnr(output_cpu, gt_cpu)
                PSNR = errs['PSNR'].item()
                if config.flag_HD_in:
                    output_cpu_ = cv2.resize(output_cpu, dsize=(0, 0), fx=1/config.scale, fy=1/config.scale, interpolation=cv2.INTER_CUBIC)
                else:
                    output_cpu_ = output_cpu

                h, w, c = output_cpu_.shape

                SSIM = ssim(output_cpu_, gt_cpu)

        PSNR_mean = PSNR_mean + PSNR
        SSIM_mean = SSIM_mean + SSIM

        frame_name = inputs['frame_name'][0]
        print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, PSNR, SSIM, itr_time))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'w' if (i == 0) else 'a') as file:
            file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, PSNR, SSIM, itr_time))
            file.close()

        ## Qualitative ##
        if not config.EVAL.quantitative_only:
            inp_cpu = inp.cpu().numpy()[0].transpose(1, 2, 0)
            for iformat in ['png', 'jpg']:
            #for iformat in ['jpg']:
                frame_name_no_ext = frame_name.split('.')[0]
                save_path_deblur = os.path.join(save_path_root_deblur, iformat)
                Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

                Path(os.path.join(save_path_deblur, 'input', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                cv2.imwrite(save_file_path_deblur_input, cv2.cvtColor(inp_cpu*255, cv2.COLOR_RGB2BGR))

                Path(os.path.join(save_path_deblur, 'output', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                cv2.imwrite(save_file_path_deblur_output, cv2.cvtColor(output_cpu*255, cv2.COLOR_RGB2BGR))

                if 'gt' in inputs.keys():
                    Path(os.path.join(save_path_deblur, 'gt', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                    save_file_path_deblur_gt = os.path.join(save_path_deblur, 'gt', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                    cv2.imwrite(save_file_path_deblur_gt, cv2.cvtColor(gt_cpu*255, cv2.COLOR_RGB2BGR))

        total_itr_time_video = total_itr_time_video + itr_time
        total_norm = total_norm + 1
        frame_len_prev = inputs['frame_len'][0]

    # total average
    total_itr_time = (total_itr_time + total_itr_time_video) / total_norm
    PSNR_mean_total = (PSNR_mean_total + PSNR_mean) / total_norm
    SSIM_mean_total = (SSIM_mean_total + SSIM_mean) / total_norm

    sys.stdout.write('\n[TOTAL {}|{}] PSNR: {:.5f}  SSIM: {:.5f} ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
        file.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
        file.close()
