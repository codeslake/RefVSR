import cog
from cog import BasePredictor, BaseModel, Input
import torch

from configs.config_RefVSR_MFID_8K import get_config
from ckpt_manager import CKPT_Manager
from trainers import create_trainer

from utils import *
from data_loader.utils import load_file_list, refine_image, read_frame

from pathlib import Path
import tempfile
import cv2

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="6"

class Output(BaseModel):
    LR_input: cog.Path
    SR_output: cog.Path

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.config = get_config('RefVSR_CVPR2022', 'RefVSR_MFID_8K', 'config_RefVSR_MFID_8K')
        self.config.network = 'RefVSR'
        self.config.EVAL.is_gradio = True
        self.config.EVAL.is_replicate = True
        self.config.frame_num = 3
        self.config.center_idx = self.config.frame_num//2

        model = create_trainer(self.config)
        self.network = model.get_network().eval()
        self.network = self.network.to(self.device)

        #ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.cuda, config.max_ckpt_num, is_descending = False)
        ckpt_manager = CKPT_Manager(root_dir='', model_name='RefVSR_MFID_8K', cuda=self.config.cuda, dist=self.config.dist)
        load_state, ckpt_name = ckpt_manager.load_ckpt(self.network, abs_name = './ckpt/RefVSR_MFID_8K.pytorch')
        print(load_state)

    def crop_img(self, img):
        max_long_side = 1280
        max_short_side = 720
        h, w, c = img.shape
        long_crop = 0
        short_crop = 0

        if max(h, w) > max_long_side:
            long_crop = max(h, w) - max_long_side
        if min(h, w) > max_short_side:
            short_crop = min(h, w) - max_short_side

        if h > w:
            h_start = long_crop//2
            h_end = long_crop//2 + max_long_side
            w_start = short_crop//2
            w_end = short_crop//2 + max_short_side
        else:
            w_start = long_crop//2
            w_end = long_crop//2 + max_long_side
            h_start = short_crop//2
            h_end = short_crop//2 + max_short_side

        img = img[h_start:h_end, w_start:w_end, :]
        return img

    def predict(self,
            LR: cog.Path = Input(description="LR ultra-wide frame to super-resolve"),
            Ref: cog.Path = Input(description="Reference wide-angle frame")
    ) -> Output:
    #) -> cog.Path:
        assert str(LR).split('.')[-1] in ['png', 'jpg'], 'image should end with ".jpg" or ".png"'
        assert str(Ref).split('.')[-1] in ['png', 'jpg'], 'image should end with ".jpg" or ".png"'


        LR_cpu = self.crop_img(read_frame(str(LR)))
        Ref_cpu = self.crop_img(read_frame(str(Ref)))
        #LR_cpu = read_frame(str(LR))
        #Ref_cpu = read_frame(str(Ref))

        LR = torch.FloatTensor(refine_image(LR_cpu, 8)[None, :, :, :].transpose(0, 3, 1, 2).copy()).to(self.device)
        Ref = torch.FloatTensor(refine_image(Ref_cpu, 8)[None, :, :, :].transpose(0, 3, 1, 2).copy()).to(self.device)

        n, c, h, w = LR.size()
        LR = LR[:, None, :, :, :].expand(n, self.config.frame_num, c, h, w)
        Ref = Ref[:, None, :, :, :].expand(n, self.config.frame_num, c, h, w)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outs = self.network(LR, Ref, True, False, False)

        output = outs['result']

        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
        output_cpu = (np.flip(output_cpu, 2) * 255).astype(np.uint8)

        out_path = cog.Path(tempfile.mkdtemp()) / 'out.png'
        cv2.imwrite(str(out_path), output_cpu)
        #cv2.imwrite('./out.png', output_cpu)

        #return out_path

        input_cpu = LR.cpu().numpy()[0, 0].transpose(1, 2, 0)
        input_cpu = (np.flip(input_cpu, 2) * 255).astype(np.uint8)
        inp_path = cog.Path(tempfile.mkdtemp()) / 'inp.png'
        cv2.imwrite(str(inp_path), input_cpu)

        return Output(LR_input=inp_path, SR_output=out_path)
