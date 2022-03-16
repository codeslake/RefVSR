import numpy as np
import math
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity

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