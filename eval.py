import torch

from utils import *
from evaluation.eval_qual_quan import eval_qual_quan
from evaluation.eval_quan_FOV import eval_quan_FOV
from evaluation.eval_quan_conf_map import eval_quan_conf_map

def eval(config):
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        eval_mode = config.EVAL.eval_mode
        print(toGreen('Evaluation Mode...'))
        print(toRed('\t'+eval_mode))

        with torch.no_grad():
            if 'qual_quan' in eval_mode:
                eval_qual_quan(config)
            elif 'FOV' in eval_mode:
                eval_quan_FOV(config)
            elif 'conf' in eval_mode:
                eval_quan_conf_map(config)
