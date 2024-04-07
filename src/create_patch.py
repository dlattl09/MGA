import numpy as np
import torch
import os
import copy
import random
from functools import partial
from torch import nn
import sys
import argparse
import torch.nn.functional as F


def extract_patches(x, kernel_size, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    batch_size, channel, D, H, W = x.shape
    x= x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1]).unfold(4, kernel_size[2], stride[2])
   # print(x.shape)
    x = x.contiguous().view(batch_size, -1,channel*kernel_size[0]*kernel_size[1]*kernel_size[2])
    return x

def get_dim_blocks(dim_in, dim_kernel_size, dim_stride=1):
    dim_out = (dim_in - dim_kernel_size)//dim_stride + 1
    return dim_out

def combine_patches(x, output_shape, kernel_size, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    batch_size = x.shape[0]
    channels = output_shape[1]
    D,H,W = kernel_size#, channels, D, H, W = x.shape
    b_out, c_out, d_dim_out, h_dim_out, w_dim_out = output_shape
    d_dim_in = get_dim_blocks(d_dim_out, kernel_size[0], stride[0])
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[1], stride[1])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[2], stride[2])
   # print(d_dim_out, kernel_size[0], stride[0])
    x = x.view(b_out, channels, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])
    x = x.permute(0,1,5,2,6,7,3,4)
    x = x.contiguous().view(b_out, channels*kernel_size[0]*d_dim_in*kernel_size[1]*kernel_size[2], 
                           h_dim_in*w_dim_in)
    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out),
                                kernel_size = (kernel_size[1], kernel_size[2]),
                                stride = (stride[1], stride[2]))
    x = x.view(b_out, channels*kernel_size[0],
              d_dim_in*h_dim_out*w_dim_out)
    x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out*w_dim_out),
                                kernel_size = (kernel_size[0],1),
                                 stride = (stride[0],1))
    out = x.view(b_out, c_out, d_dim_out, h_dim_out, w_dim_out)
    return out
