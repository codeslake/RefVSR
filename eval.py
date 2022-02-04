import torch
import torchvision.utils as vutils
import torch.nn.functional as F

import os
import sys
import datetime
import time
import gc
from pathlib import Path

import numpy as np
import cv2
import math
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
import collections

from utils import *
from data_loader.utils import refine_image_pt, read_frame, load_file_list, norm
from models.utils import warp
from ckpt_manager import CKPT_Manager

from trainers import create_trainer

def mae(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def ssim(img1, img2, PIXEL_MAX = 1.0):
    return structural_similarity(img1, img2, data_range=PIXEL_MAX, multichannel=True)

def ssim_masked(img1, img2, mask, PIXEL_MAX = 1.0):
    _, s = structural_similarity(img1, img2, data_range=PIXEL_MAX, multichannel=True, full=True)
    s = s * mask
    mssim = np.sum(s)/np.sum(mask)
    return mssim

def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse_ = np.mean( (img1 - img2) ** 2 )
    return 10 * math.log10(PIXEL_MAX / mse_)

def psnr_masked(img1, img2, mask, PIXEL_MAX = 1.0):
    mse_ = np.sum( ( (img1 - img2) ** 2) * mask) / np.sum(mask)
    return 10 * math.log10(PIXEL_MAX / mse_)

def init(config, mode = 'deblur'):
    date = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')

    model = create_trainer(config)
    model.eval()
    network = model.get_network().eval()

    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num, is_descending = False)
    load_state, ckpt_name = ckpt_manager.load_ckpt(network, by_score = config.EVAL.load_ckpt_by_score, name = config.EVAL.ckpt_name, abs_name = config.EVAL.ckpt_abs_name, epoch = config.EVAL.ckpt_epoch)
    print('\nLoading checkpoint \'{}\' on model \'{}\': {}'.format(ckpt_name, config.mode, load_state))

    save_path_root = config.EVAL.LOG_DIR.save

    save_path_root_deblur = os.path.join(save_path_root, mode, ckpt_name.split('.')[0])
    save_path_root_deblur_score = save_path_root_deblur
    Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), os.path.join(save_path_root_deblur, ckpt_name))
    save_path_root_deblur = os.path.join(save_path_root_deblur, config.EVAL.data, date)
    # Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)

    return network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name

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

        # qualitative
        # create output dir for a video
        if config.EVAL.is_qual:
            for iformat in ['png', 'jpg']:
                frame_name_no_ext = frame_name.split('.')[0]
                save_path_deblur = os.path.join(save_path_root_deblur, iformat)
                Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

                Path(os.path.join(save_path_deblur, 'input', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(inp, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)

                Path(os.path.join(save_path_deblur, 'output', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(output, '{}'.format(save_file_path_deblur_output), nrow=1, padding = 0, normalize = False)

                if 'gt' in inputs.keys():
                    Path(os.path.join(save_path_deblur, 'gt', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                    save_file_path_deblur_gt = os.path.join(save_path_deblur, 'gt', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                    vutils.save_image(gt, '{}'.format(save_file_path_deblur_gt), nrow=1, padding = 0, normalize = False)

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

def eval_quan_qual(config):
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
            results = model.evaluation(inputs, is_PSNR=config.EVAL.is_quan)
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
        if config.EVAL.is_quan:
            if 'SR_UW_png' in outs.keys() or 'SR_UW' in outs.keys():
                gt = outs['HR_UW']

                # quantitative
                output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
                gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)


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

        # qualitative
        ## create output dir for a video
        if config.EVAL.is_qual:
            for iformat in ['png', 'jpg']:
            #for iformat in ['jpg']:
                frame_name_no_ext = frame_name.split('.')[0]
                save_path_deblur = os.path.join(save_path_root_deblur, iformat)
                Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

                Path(os.path.join(save_path_deblur, 'input', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(inp, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)

                Path(os.path.join(save_path_deblur, 'output', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(output, '{}'.format(save_file_path_deblur_output), nrow=1, padding = 0, normalize = False)

                if 'gt' in inputs.keys():
                    Path(os.path.join(save_path_deblur, 'gt', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                    save_file_path_deblur_gt = os.path.join(save_path_deblur, 'gt', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                    vutils.save_image(gt, '{}'.format(save_file_path_deblur_gt), nrow=1, padding = 0, normalize = False)

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

def eval_quan_conf(config):
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

        #
        # inp = vis['lr_G']
        #

        frame_name = inputs['frame_name'][0]
        print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, itr_time))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'w' if (i == 0) else 'a') as file:
            file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, itr_time))
            file.close()

        # qualitative
        ## create output dir for a video
        for iformat in ['png', 'jpg']:
        # for iformat in ['jpg']:
            frame_name_no_ext = frame_name.split('.')[0]
            save_path_deblur = os.path.join(save_path_root_deblur, iformat)
            Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

            Path(os.path.join(save_path_deblur, 'input', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
            save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
            vutils.save_image(inp, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)

            Path(os.path.join(save_path_deblur, 'output', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
            save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
            vutils.save_image(output, '{}'.format(save_file_path_deblur_output), nrow=1, padding = 0, normalize = False)

            Path(os.path.join(save_path_deblur, 'conf_map_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
            save_file_path_deblur_conf_map_norm = os.path.join(save_path_deblur, 'conf_map_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
            vutils.save_image(conf_map_norm, '{}'.format(save_file_path_deblur_conf_map_norm), nrow=1, padding = 0, normalize = False)

            if conf_map_prop_norm is not None:
                Path(os.path.join(save_path_deblur, 'conf_map_prop_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_conf_map_prop_norm = os.path.join(save_path_deblur, 'conf_map_prop_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(conf_map_prop_norm, '{}'.format(save_file_path_deblur_conf_map_prop_norm), nrow=1, padding = 0, normalize = False)

            if conf_map_prop_b_norm is not None:
                Path(os.path.join(save_path_deblur, 'conf_map_prop_b_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_conf_map_prop_b_norm = os.path.join(save_path_deblur, 'conf_map_prop_b_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(conf_map_prop_b_norm, '{}'.format(save_file_path_deblur_conf_map_prop_b_norm), nrow=1, padding = 0, normalize = False)

            if conf_map_prop_f_norm is not None:
                Path(os.path.join(save_path_deblur, 'conf_map_prop_f_norm', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_conf_map_prop_f_norm = os.path.join(save_path_deblur, 'conf_map_prop_f_norm', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(conf_map_prop_f_norm, '{}'.format(save_file_path_deblur_conf_map_prop_f_norm), nrow=1, padding = 0, normalize = False)


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

def save(config):
    mode = config.EVAL.eval_mode
    network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name = init(config, mode)
    save_path = 'ckpt/'+config.mode+'_trans.pytorch'
    torch.save(model.get_network().state_dict(), save_path)


def eval(config):
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        eval_mode = config.EVAL.eval_mode
        print(toGreen('Evaluation Mode...'))
        print(toRed('\t'+eval_mode))

        with torch.no_grad():
            if 'quan_qual' in eval_mode:
                eval_quan_qual(config)
            elif 'FOV' in eval_mode:
                eval_quan_FOV(config)
            elif 'conf' in eval_mode:
                eval_quan_conf(config)
            elif 'save' in eval_mode:
                save(config)
