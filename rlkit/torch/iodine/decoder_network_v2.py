import torch
import torch.utils.data
from rlkit.torch.iodine.physics_network import PhysicsNetwork, PhysicsNetworkMLP
from rlkit.torch.iodine.refinement_network import RefinementNetwork
from rlkit.torch.networks import Mlp
from torch import nn
from torch.autograd import Variable
from os import path as osp
from torchvision.utils import save_image
from torch.nn import functional as F, Parameter
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from rlkit.torch.pytorch_util import from_numpy
from rlkit.torch.conv_networks import CNN, DCNN, BroadcastCNN
from rlkit.torch.vae.vae_base import GaussianLatentVAE
from rlkit.torch.modules import LayerNorm2D
from rlkit.core import logger
import os
import pdb
import copy

###Maps name to a tuple (class type, lambda function defining model architecture)
Decoder_Args = dict(
    reg = ("reg",
        lambda full_rep_size: dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=80,
        input_height=80,
        input_channels=full_rep_size + 2,
        kernel_sizes=[5, 5, 5, 5],
        n_channels=[64, 64, 64, 4],
        strides=[1, 1, 1 ,1],
        paddings=[0, 0, 0, 0],
        hidden_activation=nn.ELU())
    )
)


class DecoderNetwork_V2(nn.Module):
    def __init__(
            self,
            input_width, #All the args of just for the broadcast network
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
    ):
        super().__init__()
        self.broadcast_net = BroadcastCNN(input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes,
            added_fc_input_size,
            batch_norm_conv,
            batch_norm_fc,
            init_w,
            hidden_init,
            hidden_activation,
            output_activation)
        self.input_width = input_width #Assume images are square so don't need input_height

    #Input: latents (B,R)
    #Output: mask_logits (B,1,D,D),  colors (B,3,D,D)
    def forward(self, latents):
        broadcast_ones = ptu.ones(list(latents.shape) + [self.input_width, self.input_width]).to(latents.device) #(B,R,D,D)
        decoded = self.broadcast_net(latents, broadcast_ones) #(B,4,D,D)
        colors = decoded[:, :3] #(B,3,D,D)
        mask_logits = decoded[:, 3:4] #(B,1,D,D)
        return mask_logits, colors



