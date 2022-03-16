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
from .metrics import psnr, psnr_masked, ssim, ssim_masked
from utils import *
from data_loader.utils import refine_image_pt, read_frame, load_file_list, norm

def eval_quan_FOV(config):
    mode = config.EVAL.eval_mode
    network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name = init(config, mode)

    ##
    total_norm = 0
    total_itr_time = 0
    total_itr_time_video = 0

    keys = [1, 0.9, 0.8, 0.7, 0.6, 0.5]

    # fi: inside the overlapped fov
    # fo: outside the overlapped fov
    # fr: FOV ring
    PSNR_mean_fi_total = {}
    PSNR_mean_fo_total = {}
    PSNR_mean_fr_total = {}
    SSIM_mean_fi_total = {}
    SSIM_mean_fo_total = {}
    SSIM_mean_fr_total = {}
    PSNR_mean_fi = {}
    PSNR_mean_fo = {}
    PSNR_mean_fr = {}
    SSIM_mean_fi = {}
    SSIM_mean_fo = {}
    SSIM_mean_fr = {}

    for key in keys:
        PSNR_mean_fi_total[key] = PSNR_mean_fo_total[key] = PSNR_mean_fr_total[key] = 0
        SSIM_mean_fi_total[key] = SSIM_mean_fo_total[key] = SSIM_mean_fr_total[key] = 0

        PSNR_mean_fi[key] = PSNR_mean_fo[key] = PSNR_mean_fr[key] = 0
        SSIM_mean_fi[key] = SSIM_mean_fo[key] = SSIM_mean_fr[key] = 0

    if config.EVAL.is_debug:
        frame_count = 0

    frame_len_prev = 0
    for i, inputs in enumerate(model.data_loader_eval):
        is_first_frame = inputs['is_first'][0].item()

        # for k in inputs.keys():
        #     print(k, inputs[k].size())
        if 'is_continue' in inputs.keys() and inputs['is_continue'][0].item():
            print('passing, video', inputs['video_name'][0])
            frame_len_prev += 1
            continue


        if config.EVAL.is_debug:
            frame_count+=1
            if frame_count == 3:
                frame_len_prev = frame_count-1
                is_first_frame = True

        if is_first_frame:
            if i > 0:
                total_itr_time = total_itr_time + total_itr_time_video
                total_itr_time_video = total_itr_time_video / frame_len_prev

                for key in keys:
                    PSNR_mean_fi_total[key] += PSNR_mean_fi[key]
                    PSNR_mean_fo_total[key] += PSNR_mean_fo[key]
                    PSNR_mean_fr_total[key] += PSNR_mean_fr[key]

                    SSIM_mean_fi_total[key] += SSIM_mean_fi[key]
                    SSIM_mean_fo_total[key] += SSIM_mean_fo[key]
                    SSIM_mean_fr_total[key] += SSIM_mean_fr[key]

                    PSNR_mean_fi[key] = PSNR_mean_fi[key] / frame_len_prev
                    PSNR_mean_fo[key] = PSNR_mean_fo[key] / frame_len_prev
                    PSNR_mean_fr[key] = PSNR_mean_fr[key] / frame_len_prev
                    SSIM_mean_fi[key] = SSIM_mean_fi[key] / frame_len_prev
                    SSIM_mean_fo[key] = SSIM_mean_fo[key] / frame_len_prev
                    SSIM_mean_fr[key] = SSIM_mean_fr[key] / frame_len_prev

                out_str = '[MEAN EVAL {}|{}|{}][{}/{}] ({:.5f}sec) \n[PSNR-FOV_in  ] ('.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0], inputs['video_len'][0], total_itr_time_video)
                for k, v in PSNR_mean_fi.items():
                    out_str += '0-{:3.1f}%: {:.5f}, '.format(k*100, v)
                out_str += ')\n[PSNR-FOV_out ] ('
                for k, v in PSNR_mean_fo.items():
                    out_str += '{:3.1f}-100%: {:.5f}, '.format(k*100, v)
                out_str += ')\n[PSNR-FOV_ring] ('
                for k, v in PSNR_mean_fr.items():
                    out_str += '{:3.1f}-{:3.1f}%: {:.5f}, '.format(keys[-1]*100, k*100, v)
                out_str += ')\n[SSIM-FOV_in  ] ('
                for k, v in SSIM_mean_fi.items():
                    out_str += '0-{:3.1f}%: {:.5f}, '.format(k*100, v)
                out_str += ')\n[SSIM-FOV_out ] ('
                for k, v in SSIM_mean_fo.items():
                    out_str += '{:3.1f}-100%: {:.5f}, '.format(k*100, v)
                out_str += ')\n[SSIM-FOV_ring] ('
                for k, v in SSIM_mean_fr.items():
                    out_str += '{:3.1f}-{:3.1f}%: {:.5f}, '.format(keys[-1]*100, k*100, v)
                out_str += ') \n\n'
                print(out_str)
                if not config.EVAL.is_debug:
                    with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
                        file.write(out_str)
                        file.close()

            if config.EVAL.is_debug and frame_count == 2:
                break

            total_itr_time_video = 0
            for key in keys:
                PSNR_mean_fi[key] = PSNR_mean_fo[key] = PSNR_mean_fr[key] = 0
                SSIM_mean_fi[key] = SSIM_mean_fo[key] = SSIM_mean_fr[key] = 0

        #########################
        init_time = time.time()
        results = model.evaluation(inputs)
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

        gt = outs['HR_UW']
        # quantitative
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
        gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)

        if config.flag_HD_in:
            output_cpu_ = cv2.resize(output_cpu, dsize=(0, 0), fx=1/config.scale, fy=1/config.scale, interpolation=cv2.INTER_CUBIC)
        else:
            output_cpu_ = output_cpu

        h, w, c = output_cpu_.shape
        for key in keys:
            #if key == 1.:

            if key == 1.:
                mask_fi = np.ones_like(output_cpu_)
                PSNR_fi = psnr(output_cpu_, gt_cpu)
                SSIM_fi = ssim(output_cpu_, gt_cpu)

                PSNR_fo = 0
                SSIM_fo = 0
            else:
                crop_ratio = int(1/((1-key)/2))
                mask_fi = np.zeros_like(output_cpu_)
                mask_fi[h//crop_ratio:h-h//crop_ratio, w//crop_ratio:w-w//crop_ratio] = 1.
                PSNR_fi = psnr_masked(output_cpu_, gt_cpu, mask_fi)
                SSIM_fi = ssim_masked(output_cpu_, gt_cpu, mask_fi)

                mask_fo = np.ones_like(output_cpu_)
                mask_fo[h//crop_ratio:h-h//crop_ratio, w//crop_ratio:w-w//crop_ratio] = 0.
                PSNR_fo = psnr_masked(output_cpu_, gt_cpu, mask_fo)
                SSIM_fo = ssim_masked(output_cpu_, gt_cpu, mask_fo)

            if key > 0.5:
                mask_fr = mask_fi.copy()
                mask_fr[h//4:h-h//4, w//4:w-w//4] = 0.

                PSNR_fr = psnr_masked(output_cpu_, gt_cpu, mask_fr)
                SSIM_fr = ssim_masked(output_cpu_, gt_cpu, mask_fr)
            else:
                PSNR_fr = SSIM_fr = 0

            PSNR_mean_fi[key] += PSNR_fi
            PSNR_mean_fo[key] += PSNR_fo
            PSNR_mean_fr[key] += PSNR_fr
            SSIM_mean_fi[key] += SSIM_fi
            SSIM_mean_fo[key] += SSIM_fo
            SSIM_mean_fr[key] += SSIM_fr

            if key == 1.:
                frame_name = inputs['frame_name'][0]
                print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, PSNR_fi, SSIM_fi, itr_time))
                if config.EVAL.is_debug is False:
                    with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'w' if (i == 0) else 'a') as file:
                        file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, PSNR_fi, SSIM_fi, itr_time))
                        file.close()


            if config.EVAL.is_debug:
                print(key, 'raw:', PSNR_fi, PSNR_fo, PSNR_fr, SSIM_fi, SSIM_fo, SSIM_fr)
                print(key, 'sum:', PSNR_mean_fi[key], PSNR_mean_fo[key], PSNR_mean_fr[key], SSIM_mean_fi[key], SSIM_mean_fo[key], SSIM_mean_fr[key], '\n')

        # # qualitative
        # ## create output dir for a video
        # if not config.EVAL.quantitative_only:
        #     inp_cpu = inp.cpu().numpy()[0].transpose(1, 2, 0)
        #     for iformat in ['png', 'jpg']:
        #     #for iformat in ['jpg']:
        #         frame_name_no_ext = frame_name.split('.')[0]
        #         save_path_deblur = os.path.join(save_path_root_deblur, iformat)
        #         Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

        #         Path(os.path.join(save_path_deblur, 'input', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
        #         save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
        #         cv2.imwrite(save_file_path_deblur_input, cv2.cvtColor(inp_cpu*255, cv2.COLOR_RGB2BGR))

        #         Path(os.path.join(save_path_deblur, 'output', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
        #         save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
        #         cv2.imwrite(save_file_path_deblur_output, cv2.cvtColor(output_cpu*255, cv2.COLOR_RGB2BGR))

        #         if 'gt' in inputs.keys():
        #             Path(os.path.join(save_path_deblur, 'gt', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
        #             save_file_path_deblur_gt = os.path.join(save_path_deblur, 'gt', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
        #             cv2.imwrite(save_file_path_deblur_gt, cv2.cvtColor(gt_cpu*255, cv2.COLOR_RGB2BGR))

        total_itr_time_video = total_itr_time_video + itr_time
        total_norm = total_norm + 1
        frame_len_prev = inputs['frame_len'][0]

    # total average
    total_itr_time = (total_itr_time + total_itr_time_video) / total_norm

    for key in keys:
        PSNR_mean_fi_total[key] = (PSNR_mean_fi_total[key] + PSNR_mean_fi[key]) / total_norm
        PSNR_mean_fo_total[key] = (PSNR_mean_fo_total[key] + PSNR_mean_fo[key]) / total_norm
        PSNR_mean_fr_total[key] = (PSNR_mean_fr_total[key] + PSNR_mean_fr[key]) / total_norm
        SSIM_mean_fi_total[key] = (SSIM_mean_fi_total[key] + SSIM_mean_fi[key]) / total_norm
        SSIM_mean_fo_total[key] = (SSIM_mean_fo_total[key] + SSIM_mean_fo[key]) / total_norm
        SSIM_mean_fr_total[key] = (SSIM_mean_fr_total[key] + SSIM_mean_fr[key]) / total_norm

    out_str = '\n[TOTAL {}|{}] \n[PSNR-FOV_in  ] ('.format(ckpt_name, config.EVAL.data)
    for k, v in PSNR_mean_fi_total.items():
        out_str += '0-{:3.1f}%: {:.5f}, '.format(k*100, v)
    out_str += ')\n[PSNR-FOV_out ] ('
    for k, v in PSNR_mean_fo_total.items():
        out_str += '{:3.1f}-100%: {:.5f}, '.format(k*100, v)
    out_str += ')\n[PSNR-FOV_ring] ('
    for k, v in PSNR_mean_fr_total.items():
        out_str += '{:3.1f}-{:3.1f}%: {:.5f}, '.format(keys[-1]*100, k*100, v)

    out_str += ')\n[SSIM-FOV_in  ] ('
    for k, v in SSIM_mean_fi_total.items():
        out_str += '0-{:3.1f}%: {:.5f}, '.format(k*100, v)
    out_str += ')\n[SSIM-FOV_out ] ('
    for k, v in SSIM_mean_fo_total.items():
        out_str += '{:3.1f}-100%: {:.5f}, '.format(k*100, v)
    out_str += ')\n[SSIM-FOV_ring] ('
    for k, v in SSIM_mean_fr_total.items():
        out_str += '{:3.1f}-{:3.1f}%: {:.5f}, '.format(keys[-1]*100, k*100, v)
    out_str += ') ({:.5f}sec)\n\n'.format(total_itr_time)

    sys.stdout.write(out_str)
    if not config.EVAL.is_debug:
        with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
            file.write(out_str)
            file.close()