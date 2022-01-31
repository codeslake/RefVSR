import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data

from data_loader.utils import *

class Train_datasets(data.Dataset):
    def __init__(self, config):
        super(Train_datasets, self).__init__()
        self.config = config
        self.is_use_T = config.is_use_T
        self.patch_size = config.patch_size
        self.frame_num = config.frame_num
        self.frame_half = self.frame_num // 2
        self.scale = config.scale
        self.flag_HD_in = config.flag_HD_in
        if self.config.dist:
            self.rank = torch.distributed.get_rank()

        self.LR_UW_folder_path_list, self.LR_UW_file_path_list, _ = load_file_list(os.path.join(config.LR_data_path, config.UW_path))
        self.LR_REF_W_folder_path_list, self.LR_REF_W_file_path_list, _ = load_file_list(os.path.join(config.LR_data_path, config.W_path))
        self.LR_REF_T_folder_path_list, self.LR_REF_T_file_path_list, _ = load_file_list(os.path.join(config.LR_data_path, config.T_path))

        self.HR_UW_folder_path_list, self.HR_UW_file_path_list, _ = load_file_list(os.path.join(config.HR_data_path, config.UW_path))
        self.HR_REF_W_folder_path_list, self.HR_REF_W_file_path_list, _ = load_file_list(os.path.join(config.HR_ref_data_W_path, config.W_path))
        self.HR_REF_T_folder_path_list, self.HR_REF_T_file_path_list, _ = load_file_list(os.path.join(config.HR_ref_data_T_path, config.T_path))

        self.frame_itr_num = config.frame_itr_num

        self._init_idx()
        self.len = int(np.ceil(len(self.idx_frame_flat)))

    def _init_idx(self):
        self.idx_video = []
        self.idx_frame_flat = []
        self.idx_frame = []
        for i in range(len(self.LR_UW_file_path_list)):
            total_frames = len(self.LR_UW_file_path_list[i])

            idx_frame_temp = list(range(0, total_frames - self.frame_itr_num + 1, self.frame_itr_num))

            self.idx_frame_flat.append(idx_frame_temp)
            self.idx_frame.append(idx_frame_temp)

            for j in range(len(idx_frame_temp)):
                self.idx_video.append(i)

        self.idx_frame_flat = sum(self.idx_frame_flat, [])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video_idx = self.idx_video[index]
        frame_offset = self.idx_frame_flat[index] - self.frame_half
        LR_UW_file_path = self.LR_UW_file_path_list[video_idx]
        LR_REF_W_file_path = self.LR_REF_W_file_path_list[video_idx]
        LR_REF_T_file_path = self.LR_REF_T_file_path_list[video_idx]
        HR_UW_file_path = self.HR_UW_file_path_list[video_idx]
        HR_REF_W_file_path = self.HR_REF_W_file_path_list[video_idx]
        HR_REF_T_file_path = self.HR_REF_T_file_path_list[video_idx]

        sampled_frame_idx = np.arange(frame_offset, frame_offset + self.frame_num + self.frame_itr_num - 1)
        sampled_frame_idx = sampled_frame_idx.clip(min = self.idx_frame_flat[index], max = len(LR_UW_file_path) - 1)

        LR_UW_patches_temp = [None] * len(sampled_frame_idx)
        LR_REF_W_patches_temp = [None] * len(sampled_frame_idx)
        LR_REF_T_patches_temp = [None] * len(sampled_frame_idx)
        HR_UW_patches_temp = [None] * len(sampled_frame_idx)
        HR_REF_W_patches_temp = [None] * len(sampled_frame_idx)
        HR_REF_T_patches_temp = [None] * len(sampled_frame_idx)

        norm_val = None
        flip_val = None
        rotate_val = None
        gauss = None
        gamma = 0
        sat_factor = None
        # gamma = random.randint(0, 2)
        # sat_factor = 1 + (0.2 - 0.4*np.random.rand())

        if random.uniform(0, 1) <= 0.5:
            ran = random.uniform(0, 1)
            if ran  <= 0.3:
                rotate_val = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif ran  <= 0.6:
                rotate_val = cv2.ROTATE_90_CLOCKWISE
            else:
                rotate_val = cv2.ROTATE_180

        if random.uniform(0, 1) <= 0.5:
            ran = random.uniform(0, 1)
            if ran <= 0.3:
                flip_val = 0
            elif ran <= 0.6:
                flip_val = 1
            else:
                flip_val = -1


        for frame_idx in range(len(sampled_frame_idx)):
            sampled_idx = sampled_frame_idx[frame_idx]

            assert get_folder_name(str(Path(LR_UW_file_path[sampled_idx]))) == get_folder_name(str(Path(LR_REF_W_file_path[sampled_idx]))) == get_folder_name(str(Path(LR_REF_T_file_path[sampled_idx]))) == get_folder_name(str(Path(HR_UW_file_path[sampled_idx]))) == get_folder_name(str(Path(HR_REF_W_file_path[sampled_idx]))) == get_folder_name(str(Path(HR_REF_T_file_path[sampled_idx])))
            assert get_base_name(LR_UW_file_path[sampled_idx]) == get_base_name(LR_REF_W_file_path[sampled_idx]) == get_base_name(LR_REF_T_file_path[sampled_idx]) == get_base_name(HR_UW_file_path[sampled_idx]) == get_base_name(HR_REF_W_file_path[sampled_idx]) == get_base_name(HR_REF_T_file_path[sampled_idx])

            gauss = None
            # gauss = True if random.uniform(0, 1) <= 0.5 else None
            LR_UW_patches_temp[frame_idx] = read_frame(LR_UW_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor)
            LR_REF_W_patches_temp[frame_idx] = read_frame(LR_REF_W_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor)
            LR_REF_T_patches_temp[frame_idx] = read_frame(LR_REF_T_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor)
            #LR_REF_T_patches_temp[frame_idx] = cv2.resize(read_frame(LR_REF_T_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor), dsize=(0, 0), fx=4/5, fy=4/5, interpolation=cv2.INTER_CUBIC)

            HR_UW_patches_temp[frame_idx] = read_frame(HR_UW_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor)
            HR_REF_W_patches_temp[frame_idx] = read_frame(HR_REF_W_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor)
            HR_REF_T_patches_temp[frame_idx] = read_frame(HR_REF_T_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor)
            #HR_REF_T_patches_temp[frame_idx] = cv2.resize(read_frame(HR_REF_T_file_path[sampled_idx], norm_val, rotate_val, flip_val, gauss, gamma, sat_factor), dsize=(0, 0), fx=4/5, fy=4/5, interpolation=cv2.INTER_CUBIC)

        LR_UW_patches = np.concatenate(LR_UW_patches_temp[:len(sampled_frame_idx)], axis = 2)
        LR_REF_W_patches = np.concatenate(LR_REF_W_patches_temp[:len(sampled_frame_idx)], axis = 2)
        LR_REF_T_patches = np.concatenate(LR_REF_T_patches_temp[:len(sampled_frame_idx)], axis = 2)
        HR_UW_patches = np.concatenate(HR_UW_patches_temp[:len(sampled_frame_idx)], axis = 2)
        HR_REF_W_patches = np.concatenate(HR_REF_W_patches_temp[:len(sampled_frame_idx)], axis = 2)
        HR_REF_T_patches = np.concatenate(HR_REF_T_patches_temp[:len(sampled_frame_idx)], axis = 2)

        # LR_UW_patches, REF_W_patches, REF_T_patches, HR_UW_patches = get_patch(LR_UW_patches, REF_W_patches, REF_T_patches, HR_UW_patches, patch_size=self.patch_size, scale = self.scale)
        if self.is_use_T: # This is False
            LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches, HR_REF_W_patches, HR_REF_T_patches =\
                get_patch_T(LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches, HR_REF_W_patches, HR_REF_T_patches, patch_size=self.patch_size, scale=self.scale, flag_HD_in=self.flag_HD_in)
        else:
            LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches, HR_REF_W_patches =\
                get_patch(LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches, HR_REF_W_patches, patch_size=self.patch_size, scale=self.scale, flag_HD_in=self.flag_HD_in)

        is_first = True
        if self.idx_video[index] == self.idx_video[index - 1]:
            is_first = False

        return {'LR_UW': LR_UW_patches,
                'LR_REF_W': LR_REF_W_patches,
                'LR_REF_T': LR_REF_T_patches,
                'HR_UW': HR_UW_patches,
                'HR_REF_W': HR_REF_W_patches,
                'HR_REF_T': HR_REF_T_patches if self.is_use_T else HR_REF_W_patches,
                'is_first': is_first
               }

class Test_datasets(data.Dataset):
    def __init__(self, config, is_valid=False):
        super(Test_datasets, self).__init__()
        self.config = config
        self.is_use_T = config.is_use_T
        self.patch_size = config.patch_size
        self.frame_num = config.frame_num
        self.frame_half = self.frame_num // 2
        self.is_crop = config.is_crop
        self.scale = config.scale
        self.flag_HD_in = config.flag_HD_in
        try:
            self.vid_name = config.EVAL.vid_name
        except:
            self.vid_name = None

        self.is_valid = is_valid
        if self.config.dist:
            self.rank = torch.distributed.get_rank()
        ## validataion
        if is_valid is True:
            LR_data_path = config.VAL.LR_data_path
            HR_data_path = config.VAL.HR_data_path
            HR_ref_data_W_path = config.VAL.HR_ref_data_W_path
            HR_ref_data_T_path = config.VAL.HR_ref_data_T_path
        else:
            LR_data_path = config.EVAL.LR_data_path
            HR_data_path = config.EVAL.HR_data_path
            HR_ref_data_W_path = config.EVAL.HR_ref_data_W_path
            HR_ref_data_T_path = config.EVAL.HR_ref_data_T_path

        self.LR_UW_folder_path_list, self.LR_UW_file_path_list, _ = load_file_list(os.path.join(LR_data_path, config.UW_path))
        self.LR_REF_W_folder_path_list, self.LR_REF_W_file_path_list, _ = load_file_list(os.path.join(LR_data_path, config.W_path))
        self.LR_REF_T_folder_path_list, self.LR_REF_T_file_path_list, _ = load_file_list(os.path.join(LR_data_path, config.T_path))

        self.HR_UW_folder_path_list, self.HR_UW_file_path_list, _ = load_file_list(os.path.join(HR_data_path, config.UW_path))
        self.HR_REF_W_folder_path_list, self.HR_REF_W_file_path_list, _ = load_file_list(os.path.join(HR_ref_data_W_path, config.W_path))
        self.HR_REF_T_folder_path_list, self.HR_REF_T_file_path_list, _ = load_file_list(os.path.join(HR_ref_data_T_path, config.T_path))

        self.frame_itr_num = 1
        self._init_idx()

        idx_frame_acc = self.idx_frame.copy()
        lenHRh = 0
        for i in range(1, len(idx_frame_acc)):
            lenHRh = lenHRh + len(idx_frame_acc[i-1])
            temp = (np.array(idx_frame_acc[i]) + lenHRh).tolist()
            idx_frame_acc[i] = temp
        self.idx_frame_acc = idx_frame_acc

        self.len = int(np.ceil(len(self.idx_frame_flat)))

    def _init_idx(self):
        self.idx_video = []
        self.idx_frame_flat = []
        self.idx_frame = []
        for i in range(len(self.LR_UW_file_path_list)):
            total_frames = len(self.LR_UW_file_path_list[i])

            idx_frame_temp = list(range(0, total_frames - self.frame_itr_num + 1))

            self.idx_frame_flat.append(idx_frame_temp)
            self.idx_frame.append(idx_frame_temp)

            for j in range(len(idx_frame_temp)):
                self.idx_video.append(i)

        self.idx_frame_flat = sum(self.idx_frame_flat, [])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video_idx = self.idx_video[index]
        frame_offset = self.idx_frame_flat[index] - self.frame_half
        LR_UW_file_path = self.LR_UW_file_path_list[video_idx]
        LR_REF_W_file_path = self.LR_REF_W_file_path_list[video_idx]
        LR_REF_T_file_path = self.LR_REF_T_file_path_list[video_idx]
        HR_UW_file_path = self.HR_UW_file_path_list[video_idx]
        # HR_REF_W_file_path = self.HR_REF_W_file_path_list[video_idx]
        # HR_REF_T_file_path = self.HR_REF_T_file_path_list[video_idx]


        sampled_frame_idx = np.arange(frame_offset, frame_offset + self.frame_num + self.frame_itr_num - 1)
        sampled_frame_idx = sampled_frame_idx.clip(min = 0, max = len(LR_UW_file_path) - 1)

        ##
        video_name = LR_UW_file_path[sampled_frame_idx[self.frame_half]].split(os.sep)[-2]
        ##
        ## for evaluation (for evaluating specific video)
        if self.vid_name is not None and video_name not in self.vid_name:
            return {'is_continue': True, 'is_first': True, 'video_name': video_name}

        LR_UW_patches_temp = [None] * len(sampled_frame_idx)
        LR_REF_W_patches_temp = [None] * len(sampled_frame_idx)
        LR_REF_T_patches_temp = [None] * len(sampled_frame_idx)
        HR_UW_patches_temp = [None] * len(sampled_frame_idx)
        # HR_REF_W_patches_temp = [None] * len(sampled_frame_idx)
        # HR_REF_T_patches_temp = [None] * len(sampled_frame_idx)

        for frame_idx in range(len(sampled_frame_idx)):
            sampled_idx = sampled_frame_idx[frame_idx]

            assert get_folder_name(str(Path(LR_UW_file_path[sampled_idx]))) == get_folder_name(str(Path(LR_REF_W_file_path[sampled_idx]))) == get_folder_name(str(Path(LR_REF_T_file_path[sampled_idx]))) == get_folder_name(str(Path(HR_UW_file_path[sampled_idx])))
            assert get_base_name(LR_UW_file_path[sampled_idx]) == get_base_name(LR_REF_W_file_path[sampled_idx]) == get_base_name(LR_REF_T_file_path[sampled_idx])

            if self.config.is_crop_valid == True and self.is_valid:
                if self.flag_HD_in is False:
                    crop_offset = self.config.crop_valid_offset
                    LR_UW_patches_temp[frame_idx] = read_frame(LR_UW_file_path[sampled_idx])[crop_offset:-crop_offset, crop_offset:-crop_offset, :]
                    LR_REF_W_patches_temp[frame_idx] = read_frame(LR_REF_W_file_path[sampled_idx])[crop_offset:-crop_offset, crop_offset:-crop_offset, :]
                    LR_REF_T_patches_temp[frame_idx] = read_frame(LR_REF_T_file_path[sampled_idx])[crop_offset:-crop_offset, crop_offset:-crop_offset, :]
                    HR_UW_patches_temp[frame_idx] = read_frame(HR_UW_file_path[sampled_idx])[4*crop_offset:-4*crop_offset, 4*crop_offset:-4*crop_offset, :]
                elif self.flag_HD_in:
                    LR_UW_patches_temp[frame_idx] = read_frame(LR_UW_file_path[sampled_idx])[256:256+256*2, 480:480+480*2, :]
                    LR_REF_W_patches_temp[frame_idx] = read_frame(LR_REF_W_file_path[sampled_idx])
                    LR_REF_T_patches_temp[frame_idx] = read_frame(LR_REF_T_file_path[sampled_idx])
                    HR_UW_patches_temp[frame_idx] = read_frame(HR_UW_file_path[sampled_idx])[256:256+256*2, 480:480+480*2, :]
            else:
                LR_UW_patches_temp[frame_idx] = read_frame(LR_UW_file_path[sampled_idx])
                LR_REF_W_patches_temp[frame_idx] = read_frame(LR_REF_W_file_path[sampled_idx])
                LR_REF_T_patches_temp[frame_idx] = read_frame(LR_REF_T_file_path[sampled_idx])
                HR_UW_patches_temp[frame_idx] = read_frame(HR_UW_file_path[sampled_idx])

        LR_UW_patches = np.concatenate(LR_UW_patches_temp[:len(sampled_frame_idx)], axis = 2)
        LR_REF_W_patches = np.concatenate(LR_REF_W_patches_temp[:len(sampled_frame_idx)], axis = 2)
        LR_REF_T_patches = np.concatenate(LR_REF_T_patches_temp[:len(sampled_frame_idx)], axis = 2)
        HR_UW_patches = np.concatenate(HR_UW_patches_temp[:len(sampled_frame_idx)], axis = 2)

        if self.is_use_T: # This is currently False
            LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches =\
                get_patch_T(LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches, None , None , is_crop=False, scale=self.scale, flag_HD_in=self.flag_HD_in, is_train=False)
        else:
            LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches, _ =\
                get_patch(LR_UW_patches, LR_REF_W_patches, LR_REF_T_patches, HR_UW_patches, None , is_crop=False, scale=self.scale, flag_HD_in=self.flag_HD_in, is_train=False)

        is_first = True
        if self.idx_video[index] == self.idx_video[index - 1]:
            is_first = False


        return {'LR_UW': LR_UW_patches,
                'LR_REF_W':LR_REF_W_patches,
                'LR_REF_T':LR_REF_T_patches,
                'HR_UW': HR_UW_patches,
                'HR_REF_W': HR_UW_patches,
                'HR_REF_T': HR_UW_patches,
                #
                'is_first': is_first,
                'video_len': len(self.LR_UW_file_path_list),
                'frame_len': len(self.LR_UW_file_path_list[video_idx]),
                'video_idx': video_idx,
                'frame_idx': sampled_frame_idx[self.frame_half],
                'video_name': video_name,
                'frame_name': os.path.basename(LR_UW_file_path[sampled_frame_idx[self.frame_half]])}



