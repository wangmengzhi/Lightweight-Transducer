"""Parameter initialization."""

import math
import torch.nn as nn

def init_like_transformer_xl(n, p, std):
    """Initialize like TransformerXL.
        See https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/train.py
    Args:
        n (str): parameter name
        p (Tensor): parameter
        str (float): standard deviation
    """
    if 'norm' in n and 'weight' in n:
        assert p.dim() == 1
        nn.init.normal_(p, mean=1.0, std=std)  # layer normalization
    elif p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
    elif p.dim() == 2:
        nn.init.normal_(p, mean=0, std=std)
    else:
        raise ValueError(n)


def init_with_xavier_uniform(n, p):
    """Initialize with Xavier uniform distribution.
    Args:
        n (str): parameter name
        p (Tensor): parameter
    """
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
    elif p.dim() in [2, 3, 4]:
        nn.init.xavier_uniform_(p)  # linear layer
    else:
        raise ValueError(n)


def init_with_lecun_normal(n, p, param_init):
    """Initialize with Lecun style.
    Args:
        n (str): parameter name
        p (Tensor): parameter
        param_init (float):
    """
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
    elif p.dim() == 2:
        fan_in = p.size(1)
        nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # linear weight
    elif p.dim() == 3:
        fan_in = p.size(1) * p[0][0].numel()
        nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # 1d conv weight
    elif p.dim() == 4:
        fan_in = p.size(1) * p[0][0].numel()
        nn.init.normal_(p, mean=0., std=1. / math.sqrt(fan_in))  # 2d conv weight
    else:
        raise ValueError(n)


def init_with_uniform(n, p, param_init):
    """Initialize with uniform distribution.
    Args:
        n (str): parameter name
        p (Tensor): parameter
        param_init (float):
    """
    if p.dim() == 1:
        nn.init.constant_(p, 0.)  # bias
    elif p.dim() in [2, 3, 4]:
        nn.init.uniform_(p, a=-param_init, b=param_init)
    else:
        raise ValueError(n)
