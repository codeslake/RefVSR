import torch
import torch.nn as nn
import scipy.ndimage
import numpy as np
class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
                                 nn.ReflectionPad2d(2), 
                                 nn.Conv2d(3, 3, 3, stride=1, padding=0, bias=None, groups=3)
                                )

        self.weights_init()
    def forward(self, x):
       # for name, f in self.named_parameters():
       #     print('name:',name)
       #     print('f:',f,f.shape)
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((3,3))
        n[1,1] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=1)
        for name, f in self.named_parameters():
            weight_torch=torch.from_numpy(k)
            f.data.copy_(weight_torch)
            f.requires_grad=False
