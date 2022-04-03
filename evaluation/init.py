import torch
import datetime
import os
from pathlib import Path

from trainers import create_trainer
from ckpt_manager import CKPT_Manager

def init(config, mode = 'deblur'):
    date = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')

    model = create_trainer(config)
    model.eval()
    network = model.get_network().eval()

    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.cuda, config.dist, config.max_ckpt_num, is_descending = False)
    load_state, ckpt_name = ckpt_manager.load_ckpt(network, by_score = config.EVAL.load_ckpt_by_score, name = config.EVAL.ckpt_name, abs_name = config.EVAL.ckpt_abs_name, epoch = config.EVAL.ckpt_epoch)
    print('\nLoading checkpoint \'{}\' on model \'{}\': {}'.format(ckpt_name, config.mode, load_state))

    save_path_root = config.EVAL.LOG_DIR.save

    if config.EVAL.is_gradio is False:
        save_path_root_deblur = os.path.join(save_path_root, mode, ckpt_name.split('.')[0])
        save_path_root_deblur_score = save_path_root_deblur
        Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), os.path.join(save_path_root_deblur, ckpt_name))
        save_path_root_deblur = os.path.join(save_path_root_deblur, config.EVAL.data, date)
    else:
        save_path_root_deblur = save_path_root
        save_path_root_deblur_score = save_path_root_deblur
        Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)
    # Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)

    return network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name
