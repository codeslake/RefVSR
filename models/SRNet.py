import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import importlib

from utils import toGreen, toRed
import collections
from models.utils import norm_feat_vis

class SRNet(nn.Module):
    def __init__(self, config):
        super(SRNet, self).__init__()
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.config = config
        self.device = config.device

        if self.rank <= 0: print(toRed('\tinitializing SR network'))

        lib = importlib.import_module('models.archs.{}'.format(config.network))
        self.Network = lib.Network(config)
        self.data = collections.OrderedDict()

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):

            torch.nn.init.xavier_uniform_(m.weight, gain = self.config.wi)
            # torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, self.config.win)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def init(self):
        # not working (wi are set to None)
        if self.config.wi is not None and self.config.win is not None:
            self.Network.apply(self.weights_init)
            if self.Network.FlowNet is not None:
                self.Network.FlowNet.load_ckpt(pretrained='./ckpt/SPyNet.pytorch')

    def input_constructor(self, res):
        b, f, c, h, w = res[:]

        imgs = torch.FloatTensor(np.random.randn(b, f, c, h, w)).to(self.device)
        flows = torch.FloatTensor(np.random.randn(b, f-1, c, h, w)).to(self.device)

        # return (imgs, imgs, flows, flows)
        return {'x': imgs, 'ref': imgs}

    #####################################################
    def forward(self, x, ref, is_first_frame=True, is_log=False, is_train=False):

        outs = self.Network.forward(x, ref, is_first_frame, is_log=is_log, is_train=is_train)

        return outs
