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
        input_width=64,
        input_height=64,
        input_channels=full_rep_size + 2,
        kernel_sizes=[5, 5, 5, 5, 5],
        n_channels=[32, 32, 32, 32, 4],
        strides=[1, 1, 1 ,1, 1],
        paddings=[2, 2, 2, 2, 2],
        hidden_activation=nn.ELU())
    ),
    reg_no_share = ("reg_no_share",
        lambda full_rep_size, k: dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=64,
        input_height=64,
        input_channels=full_rep_size + 2,
        kernel_sizes=[5, 5, 5, 5, 5],
        n_channels=[32, 32, 32, 32, 4],
        strides=[1, 1, 1 ,1, 1],
        paddings=[2, 2, 2, 2, 2],
        hidden_activation=nn.ELU(),
        k=k)
    )
)


######Regular decoder########
class DecoderNetwork_v2(nn.Module):
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


######No weight sharing decoder########
class DecoderNetwork_v2_No_Sharing(nn.Module):
    def __init__(self,
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
            k=None):
        super().__init__()
        if k is None:
            raise ValueError("A value of k is needed to initialize this model!")
        self.K = k
        self.models = nn.ModuleList()
        self.input_width = input_width  # Assume images are square so don't need input_height

        for i in range(self.K):
            self.models.append(DecoderNetwork_v2(
                input_width,  # All the args of just for the broadcast network
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
                output_activation,
            ))

    #Input: latents (B*K,R)
    #Output: mask_logits (B*K,1,D,D),  colors (B*K,3,D,D)
    def forward(self, latents):
        vals_mask_logits, vals_colors = [], []
        for i in range(self.K):
            vals = self.models[i](self._get_ith_input(latents, i))
            vals_mask_logits.append(vals[0])
            vals_colors.append(vals[1])

        vals_mask_logits = torch.cat(vals_mask_logits)
        vals_colors = torch.cat(vals_colors)
        return vals_mask_logits, vals_colors

    # Input: x (bs*k,*) or None, i representing which latent to pick (Sc)
    # Input: x (bs,*) or None
    def _get_ith_input(self, x, i):
        if x is None:
            return None
        x = x.view([-1, self.K] + list(x.shape[1:]))  # (bs,k,*)
        return x[:, i]





