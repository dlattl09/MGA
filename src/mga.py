from createpatch import *
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import copy
import random
from random import shuffle
from multiprocessing.pool import ThreadPool
from functools import partial
from torch import nn
from torch.autograd import Variable
from torch.nn import Module, Conv3d, Parameter
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import sys
import argparse
import torch.nn.functional as F
from torchvision import transforms
import math,  glob
from tqdm import tqdm

# multi-hop GAT layer for MGA
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, concat=True):
        super(GATLayer, self).__init__()
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.concat        = concat         # conacat = True for all layers except the output layer.
    #    print(self.in_features)
        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))) #.cuda()
        nn.init.sparse_(self.W.data, sparsity=0.1)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1))) #.cuda()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
      #  self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU()
       # self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        # Linear Transformation
        h = torch.matmul(input, self.W) # matrix multiplication
        N = h.size()[1]
        num_batch = h.size()[0]
        self.batchnorm = nn.BatchNorm1d(N).cuda()
        
      #  theta_x = h.permute(0, 2,1)
        adj = torch.cdist(h,h)
        adj = 1/torch.exp(adj)
        
        adj = F.normalize(adj,p=1,dim=2)
        
        adj_second = torch.bmm(adj,adj)
        adj_third = torch.bmm(adj, adj_second)
        adj = adj+0.8*adj_second+0.8*0.8*adj_third
        
        adj = (adj+adj.permute(0,2,1))/2
        
        diags = torch.diagonal(adj,dim1=1,dim2=2)
        
        diags_off = torch.diagonal(adj, offset=1, dim1=1,dim2=2)
##
        # Attention Mechanism

        Wh1 = torch.matmul(h, self.a[:self.out_features,:])
        Wh2 = torch.matmul(h, self.a[self.out_features:,:])
        e = Wh1 + Wh2.permute(0,2,1)
        e = self.leakyrelu(e)
    
        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        zero_vec = torch.nan_to_num(zero_vec)
        
        attention = torch.zeros(e.shape)
      
        for i in range(num_batch):
            ranges = diags[i][:-1]-diags_off[i]
            threshold = diags[i].min()-ranges.mean()
            attention[i]  = torch.where(adj[i]>threshold, e[i], zero_vec[i])
            
        #attention = torch.where(adj > 0.5, e, zero_vec)
        attention = torch.nan_to_num(attention).cuda()
        attention = self.batchnorm(attention)
        
        attention = F.softmax(attention, dim=-1)
        h_prime   = attention#torch.matmul(attention, input)
        

        return h_prime #+self.bias

# MGA module layer
class patchgraphLayer3D(nn.Module):

    def __init__(self, num_channels, patch_ratio,reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(patchgraphLayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.patch_ratio = patch_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.layer_norm = nn.LayerNorm(num_channels_reduced)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.GAT = GATLayer(2,2)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()
        ds, hs, ws = D//self.patch_ratio, H//self.patch_ratio, W//self.patch_ratio
          
        # Average along each patches
        patch_tensor = extract_patches(input_tensor, kernel_size = (ds,hs,ws), stride=min(ds,hs,ws))
       # print('patch_tesor',patch_tensor.shape)
        num_patches = patch_tensor.size()[1]
        num_patches_reduced = num_patches//8
        
        squeeze_tensor1 = self.avg_pool(patch_tensor)
        squeeze_tensor2 = self.max_pool(patch_tensor)
        squeeze_tensor2 = torch.sum(squeeze_tensor2, dim=2)[:,:,None,None]
        squeeze_tensor2 = squeeze_tensor2.view(squeeze_tensor1.shape)

        squeeze_tensor = torch.cat([squeeze_tensor1, squeeze_tensor2], dim=2)
        # [b, num_patches, 2]
        in_f = out_f = squeeze_tensor.shape[-1]
       # self.GAT = GATLayer(in_f,out_f)        

        pa = self.GAT(squeeze_tensor)
       # pa = torch.sum(pa,dim=2)[:,:,None,None]
        output = torch.matmul(pa, patch_tensor)
        out_tensor = combine_patches(output,input_tensor.size() ,kernel_size = (ds,hs,ws), stride=min(ds,hs,ws))
        output_tensor = torch.mul(input_tensor, out_tensor) #.view(batch_size, channel, D, H, W))

        return out_tensor
