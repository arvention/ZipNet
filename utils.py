import os
import torch


def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = requires_grad
    return x


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
