import os
import numpy as np
import cv2
from pathlib import Path
import collections
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F

def read_frame(path, norm_val = None, rotate_val = None, flip_val = None, gauss = None, gamma=0, sat_factor=None):
    if norm_val == (2**16-1):
        frame = cv2.imread(path, -1)
        frame = frame / norm_val
        frame = frame[...,::-1]
    else:
        # frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # frame = frame / 255.
        frame = Image.open(path)

    if gamma == 1:
        frame = TF.adjust_gamma(frame, 1)

    if sat_factor is not None:
        frame = TF.adjust_saturation(frame, sat_factor)

    frame = np.array(frame) / 255.

    if rotate_val is not None:
        frame = cv2.rotate(frame, rotate_val)
    if flip_val is not None:
        frame = cv2.flip(frame, flip_val)
    if gauss is not None:
        row,col,ch = frame.shape
        mean = 0
        gauss = np.random.normal(mean,1e-4,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)

    frame = np.clip(frame, 0, 1.0)
    return frame

def get_patch(LR_UW, LR_W, LR_T, HR_UW, HR_W=None, is_crop=True, patch_size=64, scale=4, flag_HD_in=False, is_train=True):

    grid = 4
    if is_crop:
        # LR_UW
        LR_UW_h, LR_UW_w = LR_UW.shape[:2]
        LR_UW_p = patch_size
        LR_UW_x = random.randrange(LR_UW_w//grid, (grid-1)*LR_UW_w//grid - LR_UW_p + 1 - 15)
        LR_UW_y = random.randrange(LR_UW_h//grid, (grid-1)*LR_UW_h//grid - LR_UW_p + 1 - 15)
        patch_LR_UW = LR_UW[LR_UW_y:LR_UW_y + LR_UW_p, LR_UW_x:LR_UW_x + LR_UW_p, :]

        # W
        scale_W = 2 # 59mm/30mm
        delta = random.randint(0,30)
        LR_W_p = scale_W * LR_UW_p
        LR_W_x = (LR_UW_x-LR_UW_w//grid)*scale_W+delta
        LR_W_y = (LR_UW_y-LR_UW_h//grid)*scale_W+delta
        patch_LR_W = LR_W[LR_W_y:LR_W_y+LR_W_p, LR_W_x:LR_W_x+LR_W_p,:]

        # HR
        if flag_HD_in is False:
            HR_UW_p =  scale * LR_UW_p
            HR_UW_x, HR_UW_y = scale * LR_UW_x, scale * LR_UW_y
            patch_HR_UW = HR_UW[HR_UW_y:HR_UW_y + HR_UW_p, HR_UW_x:HR_UW_x + HR_UW_p, :]

            if is_train:
                scale_W = scale//2
                HR_W_p =  scale_W * LR_W_p
                HR_W_x, HR_W_y = scale_W * LR_W_x, scale_W * LR_W_y
                patch_HR_W = HR_W[HR_W_y:HR_W_y + HR_W_p, HR_W_x:HR_W_x + HR_W_p, :]

        else:
            patch_HR_UW = patch_LR_UW
            if is_train:
                patch_HR_W = patch_LR_W

    else:
        patch_LR_UW = LR_UW
        patch_LR_W = LR_W
        patch_HR_UW = HR_UW
        if is_train:
            patch_HR_W = HR_W

    h, w = patch_LR_UW.shape[:2]
    patch_LR_UW = patch_LR_UW.reshape((h, w, -1, 3))
    patch_LR_UW = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_LR_UW, (2, 3, 0, 1))))

    h, w = patch_LR_W.shape[:2]
    patch_LR_W = patch_LR_W.reshape((h, w, -1, 3))
    patch_LR_W = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_LR_W, (2, 3, 0, 1))))

    h, w = patch_HR_UW.shape[:2]
    patch_HR_UW = patch_HR_UW.reshape((h, w, -1, 3))
    patch_HR_UW = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_HR_UW, (2, 3, 0, 1))))

    if is_train:
        h, w = patch_HR_W.shape[:2]
        patch_HR_W = patch_HR_W.reshape((h, w, -1, 3))
        patch_HR_W = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_HR_W, (2, 3, 0, 1))))
        if flag_HD_in:
            patch_HR_W = F.interpolate(patch_LR_W, scale_factor=2, mode='bicubic', align_corners=False).clamp(0, 1)

    if is_train:
        return patch_LR_UW, patch_LR_W, patch_LR_W, patch_HR_UW, patch_HR_W
    else:
        return patch_LR_UW, patch_LR_W, patch_LR_W, patch_HR_UW, None

def get_patch_T(LR_UW, LR_W, LR_T, HR_UW, HR_W=None, HR_T=None, is_crop=True, patch_size=64, scale=4, flag_HD_in=False, is_train=True):
    grid = 20 # UW: 20X20 / W: 10X10 / T: 4X4
    if is_crop:
        # LR_UW
        pad_size_UW_T = 8 # pad of UW's grid with respect to T's grid
        LR_UW_h, LR_UW_w = LR_UW.shape[:2]
        LR_UW_p = patch_size
        LR_UW_x = random.randrange(pad_size_UW_T*LR_UW_w//grid, (grid - pad_size_UW_T)*LR_UW_w//grid - LR_UW_p + 1 - 15)
        LR_UW_y = random.randrange(pad_size_UW_T*LR_UW_h//grid, (grid - pad_size_UW_T)*LR_UW_h//grid - LR_UW_p + 1 - 15)
        patch_LR_UW = LR_UW[LR_UW_y:LR_UW_y + LR_UW_p, LR_UW_x:LR_UW_x + LR_UW_p, :]

        #print('grid:', grid, 'pad size UW:', pad_size_UW_T, 'LR_UW_h:', LR_UW_h, 'LR_UW_w:', LR_UW_w, 'LR_UW_p:', LR_UW_p)

        #print('[LR_UW] LR_UW_h: {}, LR_UW_w: {}, LR_UW_p: {}, LR_UW_x: {}, LR_UW_y: {}, patch_LR_UW.size():'.format(LR_UW_h, LR_UW_w, LR_UW_p, LR_UW_x, LR_UW_y), patch_LR_UW.shape)

        # W
        scale_W = 2 # 59mm/30mm
        pad_size_UW_W = 5 # respect to W's grid
        delta = random.randint(0,30)
        LR_W_p = int(scale_W * LR_UW_p)
        LR_W_x = int((LR_UW_x-pad_size_UW_W*LR_UW_w//grid)*scale_W+delta)
        LR_W_y = int((LR_UW_y-pad_size_UW_W*LR_UW_h//grid)*scale_W+delta)
        patch_LR_W = LR_W[LR_W_y:LR_W_y+LR_W_p, LR_W_x:LR_W_x+LR_W_p,:]
        #print('[LR_W] LR_W_p: {}, LR_W_x: {}, LR_W_y: {}, LR_W.size(): {}, patch_LR_W.size(): {}'.format(LR_W_p, LR_W_x, LR_W_y, LR_W.shape, patch_LR_W.shape))

        # T 
        scale_T = 5 # 147mm/30mm
        delta = random.randint(0,60)
        LR_T_p = int(scale_T * LR_UW_p)

        LR_T_x = int((LR_UW_x-pad_size_UW_T*LR_UW_w//grid)*scale_T+delta)
        LR_T_y = int((LR_UW_y-pad_size_UW_T*LR_UW_h//grid)*scale_T+delta)
        patch_LR_T = LR_T[LR_T_y:LR_T_y+LR_T_p, LR_T_x:LR_T_x+LR_T_p,:]
        #print('[LR_T] LR_T_p: {}, LR_T_x: {}, LR_T_y: {}, LR_T.size(): {}, patch_LR_T.size(): {}'.format(LR_T_p, LR_T_x, LR_T_y, LR_T.shape, patch_LR_T.shape))

        # HR
        if flag_HD_in is False:
            HR_UW_p =  scale * LR_UW_p
            HR_UW_x, HR_UW_y = scale * LR_UW_x, scale * LR_UW_y
            patch_HR_UW = HR_UW[HR_UW_y:HR_UW_y + HR_UW_p, HR_UW_x:HR_UW_x + HR_UW_p, :]

            if is_train:
                lr_scale = 2
                HR_W_p = lr_scale * LR_W_p
                HR_W_x, HR_W_y = lr_scale * LR_W_x, lr_scale * LR_W_y
                patch_HR_W = HR_W[HR_W_y:HR_W_y + HR_W_p, HR_W_x:HR_W_x + HR_W_p, :]
                #print('[HR_W] HR_W_p: {}, HR_W_x: {}, HR_W_y: {}, HR_W.size(): {}, patch_HR_W.size(): {}'.format(HR_W_p, HR_W_x, HR_W_y, HR_W.shape, patch_HR_W.shape))

        else:
            patch_HR_UW = patch_LR_UW
            patch_HR_W = patch_LR_W
        patch_HR_T = patch_LR_T

    else:
        patch_LR_UW = LR_UW
        patch_LR_W = LR_W
        patch_LR_T = LR_T
        patch_HR_UW = HR_UW
        if is_train:
            patch_HR_W = HR_W
            patch_HR_T = HR_T

    h, w = patch_LR_UW.shape[:2]
    patch_LR_UW = patch_LR_UW.reshape((h, w, -1, 3))
    patch_LR_UW = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_LR_UW, (2, 3, 0, 1))))

    h, w = patch_LR_W.shape[:2]
    patch_LR_W = patch_LR_W.reshape((h, w, -1, 3))
    patch_LR_W = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_LR_W, (2, 3, 0, 1))))

    h, w = patch_LR_T.shape[:2]
    patch_LR_T = patch_LR_T.reshape((h, w, -1, 3))
    patch_LR_T = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_LR_T, (2, 3, 0, 1))))
    patch_LR_T = F.interpolate(patch_LR_T, scale_factor=4/5, mode='bicubic', align_corners=False).clamp(0, 1)

    h, w = patch_HR_UW.shape[:2]
    patch_HR_UW = patch_HR_UW.reshape((h, w, -1, 3))
    patch_HR_UW = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_HR_UW, (2, 3, 0, 1))))

    if is_train:
        h, w = patch_HR_W.shape[:2]
        patch_HR_W = patch_HR_W.reshape((h, w, -1, 3))
        patch_HR_W = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_HR_W, (2, 3, 0, 1))))

        h, w = patch_HR_T.shape[:2]
        patch_HR_T = patch_HR_T.reshape((h, w, -1, 3))
        patch_HR_T = torch.FloatTensor(np.ascontiguousarray(np.transpose(patch_HR_T, (2, 3, 0, 1))))
        patch_HR_T = F.interpolate(patch_HR_T, scale_factor=4/5, mode='bicubic', align_corners=False).clamp(0, 1)


    #print(patch_LR_UW.size(), patch_LR_W.size(), patch_LR_T.size(), patch_HR_UW.size(), patch_HR_W.size(), patch_HR_T.size())

    if is_train:
        return patch_LR_UW, patch_LR_W, patch_LR_T, patch_HR_UW, patch_HR_W, patch_HR_T
    else:
        return patch_LR_UW, patch_LR_W, patch_LR_T, patch_HR_UW

def norm(inp):
    return (inp + 1.) / 2.

def color_to_gray(img):
    c_linear = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.07228*img[:, :, 2]
    c_linear_temp = c_linear.copy()

    c_linear_temp[np.where(c_linear <= 0.0031308)] = 12.92 * c_linear[np.where(c_linear <= 0.0031308)]
    c_linear_temp[np.where(c_linear > 0.0031308)] = 1.055 * np.power(c_linear[np.where(c_linear > 0.0031308)], 1.0/2.4) - 0.055

    img[:, :, 0] = c_linear_temp
    img[:, :, 1] = c_linear_temp
    img[:, :, 2] = c_linear_temp

    return img

def refine_image(img, val = 16):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val]

def refine_image_pt(image, val = 16):
    size = image.size()
    if len(size) == 5:
        h = size[3]
        w = size[4]
        return image[:, :, :, :h - h % val, :w - w % val]

    elif len(size) == 4:
        h = size[2]
        w = size[3]
        return image[:, :, :h - h % val, :w - w % val]

def load_file_list(root_path, child_path = None, is_flatten=False):
    folder_paths = []
    filenames_pure = []
    filenames_structured = []
    num_files = 0
    for root, dirnames, filenames in os.walk(root_path):
        # print('root: ', root)
        # print('dirnames: ', dirnames)
        # print('filenames: ', filenames)
        if len(dirnames) != 0:
            if dirnames[0][0] == '@':
                del(dirnames[0])

        if len(dirnames) == 0:
            if root == '.':
                continue
            if child_path is not None and child_path != Path(root).name:
                continue
            folder_paths.append(root)
            filenames_pure = []
            for i in np.arange(len(filenames)):
                if filenames[i][0] != '.' and filenames[i] != 'Thumbs.db':
                    filenames_pure.append(os.path.join(root, filenames[i]))
            filenames_pure
            filenames_structured.append(np.array(sorted(filenames_pure), dtype='str'))
            num_files += len(filenames_pure)

    folder_paths = np.array(folder_paths)
    filenames_structured = np.array(filenames_structured, dtype=object)

    sort_idx = np.argsort(folder_paths)
    folder_paths = folder_paths[sort_idx]
    filenames_structured = filenames_structured[sort_idx]

    if is_flatten:
        if len(filenames_structured) > 1:
            filenames_structured = np.concatenate(filenames_structured).ravel()
        else:
            filenames_structured = filenames_structured.flatten()

    return folder_paths, filenames_structured, num_files

def get_base_name(path):
    return os.path.basename(path.split('.')[0])

def get_folder_name(path):
    path = os.path.dirname(path)
    return path.split(os.sep)[-1]

