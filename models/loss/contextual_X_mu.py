import torch
import torch.nn as nn
from .vgg import VGG19
import torch.nn.functional as F
LOSS_TYPES = ['cosine']
def contextual_loss(x=torch.Tensor,
                    y=torch.Tensor,
                    band_width= 0.5,
                    loss_type= 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """
    #print('band_width:',band_width)
    #assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W = x.size()

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
 
    dist_tilde = compute_relative_distance(dist_raw)
    cx_ = compute_cx(dist_tilde, band_width)

    r_m = torch.max(cx_, dim=1, keepdim=True)
    c = torch.gather(torch.exp((1 - dist_raw) / 0.5), 1, r_m[1])
    rank = torch.distributed.get_rank()
    cx = torch.sum(torch.squeeze(r_m[0]*c,1), dim=1) / torch.sum(torch.squeeze(c,1), dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5)) 


    c = c.view(N, 1, y.shape[2], y.shape[3])
    return cx_loss, c

def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid

def contextual_bilateral_loss(x=torch.Tensor,
                    y=torch.Tensor,
                    weight_sp= 0.1,
                    band_width= 0.5,
                    loss_type= 'cosine'):

    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W = x.size()

    grid = compute_meshgrid(x.shape).to(x.device)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == 'L2':
        dist_raw = compute_l2_distance(x, y)
    elif loss_type == 'L1':
        dist_raw = compute_l1_distance(x, y)
 
    dist_tilde = compute_relative_distance(dist_raw)
    cx_ = compute_cx(dist_tilde, band_width)

    cx_ = (1. - weight_sp) * cx_ + weight_sp * cx_sp

    r_m = torch.max(cx_, dim=1, keepdim=True)
    c = torch.gather(torch.exp((1 - dist_raw) / band_width), 1, r_m[1])
    # print('\n\n', dist_tilde.min(), dist_tilde.mean(), dist_tilde.max(), '\n\n')
    rank = torch.distributed.get_rank()
    cx = torch.sum(torch.squeeze(r_m[0]*c,1), dim=1) / torch.sum(torch.squeeze(c,1), dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5)) 

    c = c.view(N, 1, y.shape[2], y.shape[3])
    return cx_loss, c

def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / (torch.sum(w, dim=2, keepdim=True) + 1e-5)  # Eq(4)
    return cx

def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde

def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)
    A = y_vec.transpose(1, 2) @ x_vec
    # print(x.shape, y_s.shape, A.shape, x_s.shape)
    dist = y_s - 2 * A + x_s
    dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
    dist = dist.clamp(min=0.)
    return dist


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_mu = x.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - x_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # # consine similarity
    # cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
    #                        y_normalized)  # (N, H*W, H*W)

    # # convert to distance
    # dist = 1 - cosine_sim
    # dist = torch.clamp(dist, min=0)

    dist = torch.clamp(1 - torch.bmm(x_normalized.transpose(1, 2), y_normalized), min=0)


    return dist

class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.
    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width = 0.5,
                 loss_type = 'cosine',
                 is_CoBi = False,
                 use_vgg = True,
                 vgg_layer = 'relu3_4'):
        super(ContextualLoss, self).__init__()


        self.band_width = band_width
        self.is_CoBi = is_CoBi

        if use_vgg:
            # print('use_vgg:',use_vgg)
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 channel images.'
            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)


        if self.is_CoBi:
            return contextual_bilateral_loss(x, y, band_width=self.band_width)
        else:
            return contextual_loss(x, y, band_width=self.band_width)
