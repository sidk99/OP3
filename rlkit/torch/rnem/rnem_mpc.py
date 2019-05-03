import torch
import torch.utils.data
from torch import nn
# from torch.nn import functional as F
# from rlkit.pythonplusplus import identity
# from rlkit.torch import pytorch_util as ptu
# from torch.nn.modules.loss import BCEWithLogitsLoss
# import numpy as np
# from rlkit.torch.pytorch_util import from_numpy
# from rlkit.torch.conv_networks import CNN, DCNN
# from rlkit.torch.vae.vae_base import GaussianLatentVAE

import pytorch_utils as ptu


######
class RNEM_MPC(nn.Module):
    def __init__(self, saved_model, k):
        super(RNEM_MPC, self).__init__()
        self.model = saved_model
        self.representation_size = 500
        self.imsize = 64
        self.K = k

        self.refine_steps = 15
        self.seed_steps = 4

    def get_init_state(self, bs):
        h = ptu.zeros((bs*self.K, self.representation_size))

        pred_shape = [bs, self.K] + self.input_size_list #B, K, C, W, H
        pred = ptu.ones(pred_shape)

        gamma_shape = [bs, self.K, 1, self.imsize, self.imsize]#B, K, 1, W, H
        gamma = ptu.normal(0, 1).sample(torch.Size(gamma_shape))
        gamma = torch.abs(gamma) + 1e-6
        gamma /= torch.sum(gamma, dim=1, keepdim=True)  # (B, K, 1, W, H)

        d_preds = torch.abs(ptu.normal(0, 1).sample(torch.Size(gamma_shape)).to(self.device))
        return [h, pred, d_preds, gamma]

    def refine(self, goal_image_tensor, hidden_state):
        #goal_image_tensor: (bs, 3, imsize, imsize)
        bs = goal_image_tensor.size()[0]
        goal_image_tensor = goal_image_tensor.permute([0, 2, 3, 1]) #(bs, imsize, imsize, 3)
        goal_image_tensor = torch.unsqueeze(goal_image_tensor, 1) #(bs, 1, imsize, imsize, 3)

        if hidden_state is None:
            hidden_state = self.get_init_state(bs)

        for i in range(self.refine_steps):
            hidden_state = self.model.improve(goal_image_tensor, hidden_state, goal_image_tensor)
        return hidden_state[0].view((bs, self.K, -1))

    def step(self, initial_image_tensor, action):
        # initial_image_tensor: (bs, 3, imsize, imsize)
        bs = initial_image_tensor.size()[0]
        initial_image_tensor = initial_image_tensor.permute([0, 2, 3, 1])  # (bs, imsize, imsize, 3)
        initial_image_tensor = torch.unsqueeze(initial_image_tensor, 1)  # (bs, 1, imsize, imsize, 3)
        target_shape = [bs, -1, self.imsize, self.imsize, 3]

        hidden_state = self.get_init_state(bs)
        for i in range(self.refine_steps):
            hidden_state = self.model.improve(initial_image_tensor, hidden_state, initial_image_tensor)

        # First input action
        hidden_state = self.model.physics(target_shape, None, hidden_state, action)

        #Then rollout physics
        for i in range(10):
            hidden_state = self.model.physics(target_shape, None, hidden_state, None)

        #Note, gamma is equal to the depth probabilities
        recon = torch.sum(hidden_state[1] * hidden_state[3], dim=1, keepdim=True)
        return recon, hidden_state[0].view((bs, self.K, -1))


# class NEM(nn.Module):
#     def __init__(
#             self,
#             representation_size,
#             refinement_net,
#             decoder,
#             dynamics,
#             K,
#             decoder_distribution='bernoulli',
#             input_channels=1,
#             imsize=48,
#     ):
#         """
#         :param representation_size:
#         :param
#         """
#         super().__init__(representation_size)
#
#
#         self.input_channels = input_channels
#         self.imsize = imsize
#         self.imlength = self.imsize * self.imsize * self.input_channels
#         self.input_size_list = [input_channels, imsize, imsize]
#
#         self.representation_size = representation_size
#         self.K = K
#         self.refinement_net = refinement_net
#         self.decoder = decoder
#         self.dynamics = dynamics
#
#
#         self.epoch = 0
#         self.decoder_distribution = decoder_distribution
#         self.e_sigma = 0.25
#
#     def get_init_state(self, bs):
#         h = ptu.zeros((bs, self.K, self.representation_size))
#
#         pred_shape = [bs, self.K] + self.input_size_list #B, K, C, W, H
#         pred = ptu.ones(pred_shape)
#
#         gamma_shape = [bs, self.K, 1, self.imsize, self.imsize]#B, K, 1, W, H
#         gamma = ptu.normal(0, 1).sample(torch.Size(gamma_shape))
#         gamma = torch.abs(gamma) + 1e-6
#         gamma /= torch.sum(gamma, dim=1, keepdim=True)  # (B, K, 1, W, H)
#
#         d_preds = torch.abs(ptu.normal(0, 1).sample(torch.Size(gamma_shape)).to(self.device))
#         return [h, pred, d_preds, gamma]
#
#
#     def e_step(self, preds, targets, depth_preds):
#         # Initial part computes pixelwise loss of predictions in respect to target
#         # predictions: (B, K, C, W, H), data: (B, 1, C, W, H), prob: (B, K, 1, W, H)
#         if targets is None:
#             return torch.nn.functional.softmax(depth_preds, dim=1)
#
#         if self.decoder_distribution == "bernoulli":
#             mu = preds
#             prob = targets * mu + (1 - targets) * (1 - mu)  # RV: Note data is binary
#         elif self.decoder_distribution == "gaussian":
#             mu, sigma = preds, self.e_sigma
#             prob = ((1 / ptu.FloatTensor([np.sqrt((2 * np.pi * sigma ** 2))]))
#                     * torch.exp(torch.sum(-torch.pow(targets - mu, 2), dim=3, keepdim=True) / (2 * sigma ** 2)))
#             #prob = torch.sum(prob, dim=3, keepdim=True) # TODO Check if right for color
#         else:
#             raise ValueError('Unknown distribution_type: "{}"'.format(self.pixel_dist))
#
#         depth_probs = torch.nn.functional.softmax(depth_preds, dim=1)
#         gamma = depth_probs * prob + 1e-6  # RV: Computing posterior probability
#
#         gamma = gamma/torch.sum(gamma, dim=1, keepdim=True)
#         return gamma
#
#     def refine(self, images, state):
#         bs = images.size()[0]
#         #state contains: h_old, preds_old, d_preds_old, gamma_old
#         if state is None:
#             state = self.get_init_state(bs)
#         h_mstep = self.refinement_net(images, state)  #Run encoders
#         mu_preds, depth_preds = self.decoder(h_mstep)  # Run decoders
#         gamma = self.e_step(mu_preds, images, depth_preds)
#
#         state = [h_mstep, mu_preds, depth_preds, gamma]
#
#         d_probs = torch.nn.functional.softmax(depth_preds, dim=1)
#         reconstructed_image = torch.sum(mu_preds * d_probs, dim=1, keepdim=True) #No lookahead bias here
#         return reconstructed_image, state
#
#     def predict(self, target_image, state, action):
#         h = state[0]
#         h = self.dynamics(h, action)
#
#         mu_preds, depth_preds = self.decoder(h)  # Run decoders
#         gamma = self.e_step(mu_preds, target_image, depth_preds)
#
#         state = [h, mu_preds, depth_preds, gamma]
#         d_probs = torch.nn.functional.softmax(depth_preds, dim=1)
#         reconstructed_image = torch.sum(mu_preds * d_probs, dim=1, keepdim=True)  # No lookahead bias
#         return reconstructed_image, state
#
#

if __name__ == "__main__":
    filepath = "../../../../sharedGit/Relational-NEM/experiments/4BlocksActions_s1000s5t0.1n0.00k5_5-2-23_FFTFT/model.pkl"
    model = torch.load(open(filepath, 'rb'), map_location='cpu')
    model.eval()

    print(model.training_info)
    print(model)
