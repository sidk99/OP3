import torch
import torch.utils.data
from rlkit.torch.iodine.physics_network_v2 import PhysicsNetwork_v2, PhysicsNetworkMLP_v2, Physics_Args
from rlkit.torch.iodine.refinement_network_v2 import RefinementNetwork_v2, Refinement_Args
from rlkit.torch.iodine.decoder_network_v2 import DecoderNetwork_V2, Decoder_Args

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

# imsize84_iodine_architecture = dict(
#     deconv_args=dict(
#         hidden_sizes=[],
#
#         input_width=92,
#         input_height=92,
#         input_channels=130,
#
#         kernel_sizes=[3, 3, 3, 3],
#         n_channels=[64, 64, 64, 64],
#         paddings=[0, 0, 0, 0],
#         strides=[1, 1, 1, 1],
#     ),
#     deconv_kwargs=dict(
#         batch_norm_conv=False,
#         batch_norm_fc=False,
#     ),
#     refine_args=dict(
#         input_width=84,
#         input_height=84,
#         input_channels=17,
#         paddings=[0, 0, 0, 0],
#         kernel_sizes=[3, 3, 3, 3],
#         n_channels=[64, 64, 64, 64],
#         strides=[2, 2, 2, 2],
#         hidden_sizes=[128, 128],
#         output_size=128,
#         lstm_size=256,
#         lstm_input_size=768,
#         added_fc_input_size=0
#
#     )
# )
#
# imsize64_iodine_architecture = dict(
#     deconv_args=dict(
#         hidden_sizes=[],
#
#         input_width=80,
#         input_height=80,
#         input_channels=34,
#
#         kernel_sizes=[5, 5, 5, 5],
#         n_channels=[32, 32, 32, 32],
#         strides=[1, 1, 1, 1],
#         paddings=[0, 0, 0, 0]
#     ),
#     deconv_kwargs=dict(
#         batch_norm_conv=False,
#         batch_norm_fc=False,
#     ),
#     refine_args=dict(
#         input_width=64,
#         input_height=64,
#         input_channels=17,
#         paddings=[0, 0, 0, 0],
#         kernel_sizes=[5, 5, 5, 5],
#         n_channels=[32, 32, 32, 32],
#         strides=[2, 2, 2, 2],
#         hidden_sizes=[128, 128],
#         output_size=32,
#         lstm_size=128,
#         lstm_input_size=288,
#         added_fc_input_size=0
#
#     )
# )
#
# REPSIZE_128 = 128
#
# imsize64_large_iodine_architecture = dict(
#     vae_kwargs=dict(
#         imsize=64,
#         representation_size=REPSIZE_128,
#         input_channels=3,
#         # decoder_distribution='gaussian_identity_variance',
#         beta=1,
#         # K=7,
#         sigma=0.1,
#     ),
#     deconv_args=dict(
#         hidden_sizes=[],
#         output_size=64 * 64 * 3,
#         input_width=80,
#         input_height=80,
#         input_channels=REPSIZE_128 + 2,
#
#         kernel_sizes=[5, 5, 5, 5],
#         n_channels=[64, 64, 64, 64],
#         strides=[1, 1, 1, 1],
#         paddings=[0, 0, 0, 0]
#     ),
#     deconv_kwargs=dict(
#         batch_norm_conv=False,
#         batch_norm_fc=False,
#     ),
#     refine_args=dict(
#         input_width=64,
#         input_height=64,
#         input_channels=17,
#         paddings=[0, 0, 0, 0],
#         kernel_sizes=[5, 5, 5, 5],
#         n_channels=[64, 64, 64, 64],
#         strides=[2, 2, 2, 2],
#         hidden_sizes=[128, 128],
#         output_size=REPSIZE_128,
#         lstm_size=256,
#         lstm_input_size=768,
#         added_fc_input_size=0
#
#     ),
#     physics_kwargs=dict(
#         action_enc_size=32,
#     ),
#     schedule_kwargs=dict(
#         train_T=5,
#         test_T=5,
#         seed_steps=4,
#         schedule_type='single_step_physics'
#     )
# )
#
# imsize64_large_iodine_architecture_multistep_physics = dict(
#     vae_kwargs=dict(
#         imsize=64,
#         representation_size=REPSIZE_128,
#         input_channels=3,
#         # decoder_distribution='gaussian_identity_variance',
#         beta=1,
#         # K=7, #7
#         sigma=0.1,
#     ),
#     deconv_args=dict(
#         hidden_sizes=[],
#         output_size=64 * 64 * 3,
#         input_width=80,
#         input_height=80,
#         input_channels=REPSIZE_128 + 2,
#
#         kernel_sizes=[5, 5, 5, 5],
#         n_channels=[64, 64, 64, 64],
#         strides=[1, 1, 1, 1],
#         paddings=[0, 0, 0, 0]
#     ),
#     deconv_kwargs=dict(
#         batch_norm_conv=False,
#         batch_norm_fc=False,
#     ),
#     refine_args=dict(
#         input_width=64,
#         input_height=64,
#         input_channels=17,
#         paddings=[0, 0, 0, 0],
#         kernel_sizes=[5, 5, 5, 5],
#         n_channels=[64, 64, 64, 64],
#         strides=[2, 2, 2, 2],
#         hidden_sizes=[128, 128],
#         output_size=REPSIZE_128,
#         lstm_size=256,
#         lstm_input_size=768,
#         added_fc_input_size=0
#
#     ),
#     physics_kwargs=dict(
#         action_enc_size=32,
#     )  # ,
#     # schedule_kwargs=dict(
#     #     train_T=10,
#     #     test_T=10,
#     #     seed_steps=5,
#     #     schedule_type='random_alternating'
#     # )
# )
#
# imsize64_small_iodine_architecture = dict(
#     vae_kwargs=dict(
#         imsize=64,
#         representation_size=REPSIZE_128,
#         input_channels=3,
#         beta=1,
#         sigma=0.1,
#     ),
#     deconv_args=dict(
#         hidden_sizes=[],
#         output_size=64 * 64 * 4,  # Note: This does not seem to be used in Broadcast
#         input_width=64,
#         input_height=64,
#         input_channels=REPSIZE_128 + 2,
#
#         kernel_sizes=[5, 5],
#         n_channels=[64, 4],
#         strides=[1, 1],
#         paddings=[2, 2]
#     ),
#     deconv_kwargs=dict(
#         batch_norm_conv=False,
#         batch_norm_fc=False,
#     ),
#     refine_args=dict(
#         input_width=64,
#         input_height=64,
#         input_channels=17,
#         paddings=[0, 0],
#         kernel_sizes=[5, 5],
#         n_channels=[64, 64],
#         strides=[2, 2],
#         hidden_sizes=[128, 128],
#         output_size=REPSIZE_128,
#         lstm_size=256,
#         lstm_input_size=768,
#         added_fc_input_size=0
#     ),
#     physics_kwargs=dict(
#         action_enc_size=32,
#     )
# )
#
# imsize64_medium_iodine_architecture = dict(
#     vae_kwargs=dict(
#         imsize=64,
#         representation_size=REPSIZE_128,
#         input_channels=3,
#         beta=1,
#         sigma=0.1,
#     ),
#     deconv_args=dict(
#         hidden_sizes=[],
#         output_size=64 * 64 * 3,
#         input_width=64,
#         input_height=64,
#         input_channels=REPSIZE_128 + 2,
#
#         kernel_sizes=[3, 3, 3, 3],
#         n_channels=[64, 64, 64, 4],
#         strides=[1, 1, 1, 1],
#         paddings=[1, 1, 1, 1]
#     ),
#     deconv_kwargs=dict(
#         batch_norm_conv=False,
#         batch_norm_fc=False,
#     ),
#     refine_args=dict(
#         input_width=64,
#         input_height=64,
#         input_channels=17,
#         paddings=[0, 0, 0],
#         kernel_sizes=[3, 3, 3],
#         n_channels=[64, 64, 64],
#         strides=[1, 1, 1],
#         hidden_sizes=[128, 128],
#         output_size=REPSIZE_128,
#         lstm_size=256,
#         lstm_input_size=768,
#         added_fc_input_size=0
#     ),
#     physics_kwargs=dict(
#         action_enc_size=32,
#     )
# )


## schedule_parameters=dict(
##     train_T = 21,
##     test_T = 21,
##     seed_steps = 5,
##     schedule_type='random_alternating'
## )


# def create_model(variant, action_dim):
#     # pdb.set_trace()
#     # if 'K' in variant.keys(): #New version
#     K = variant['K']
#     # else: #Old version
#     #     K = variant['vae_kwargs']['K']
#     print('K: {}'.format(K))
#     model = variant['model']
#     rep_size = model['vae_kwargs']['representation_size']
#
#     decoder = BroadcastCNN(**model['deconv_args'], **model['deconv_kwargs'],
#                            hidden_activation=nn.ELU())
#     refinement_net = RefinementNetwork(**model['refine_args'],
#                                        hidden_activation=nn.ELU())
#     physics_net = PhysicsNetwork(K, rep_size, action_dim, **model['physics_kwargs'])
#
#     m = IodineVAE(
#         **model['vae_kwargs'],
#         **variant['schedule_kwargs'],
#         K=K,
#         decoder=decoder,
#         refinement_net=refinement_net,
#         physics_net=physics_net,
#         action_dim=action_dim,
#     )
#     # pdb.set_trace()
#     return m
#
#
# def create_schedule(train, T, schedule_type, seed_steps, max_T=None):
#     if schedule_type == 'single_step_physics':
#         schedule = np.ones((T,))
#         schedule[:seed_steps] = 0
#     elif schedule_type == 'random_alternating':
#         if train:
#             schedule = np.random.randint(0, 2, (T,))
#         else:
#             schedule = np.ones((T,))
#         schedule[:seed_steps] = 0
#     elif schedule_type == 'multi_step_physics':
#         schedule = np.ones((T,))
#         schedule[:seed_steps] = 0
#     elif 'curriculum' in schedule_type:
#         if train:
#             max_multi_step = int(schedule_type[-1])
#             # schedule = np.zeros((T,))
#             rollout_len = np.random.randint(max_multi_step) + 1
#             schedule = np.zeros(seed_steps + rollout_len + 1)
#             schedule[seed_steps:seed_steps + rollout_len] = 1  # schedule looks like [0,0,0,0,1,1,1,0]
#         else:
#             max_multi_step = int(schedule_type[-1])
#             schedule = np.zeros(seed_steps + max_multi_step + 1)
#             schedule[seed_steps:seed_steps + max_multi_step] = 1
#     else:
#         raise Exception
#     if max_T is not None:  # Enforces that we have at most max_T-1 physics steps
#         timestep_count = np.cumsum(schedule)
#         schedule = np.where(timestep_count <= max_T - 1, schedule, 0)
#     # print(schedule)
#     return schedule
#
#
# ####Get loss weight depending on schedule
# def get_loss_weight(t, schedule, schedule_type):
#     if schedule_type == 'single_step_physics':
#         return t
#     elif schedule_type == 'random_alternating':
#         return t
#     elif schedule_type == 'multi_step_physics':
#         return t
#     elif 'curriculum' in schedule_type:
#         return t
#     else:
#         raise Exception
#
#
# class IodineVAE(GaussianLatentVAE):
#     def __init__(
#             self,
#             representation_size,
#             refinement_net,
#             decoder,
#             action_dim=None,
#             physics_net=None,
#             K=3,
#             input_channels=1,
#             imsize=48,
#             min_variance=1e-3,
#             beta=5,
#             sigma=0.1,
#             train_T=5,
#             test_T=5,
#             seed_steps=4,
#             schedule_type='single_step_physics'
#
#     ):
#         """
#
#         :param imsize:
#         :param init_w:
#         :param min_variance:
#         :param hidden_init:
#         """
#         super().__init__(representation_size)
#         if min_variance is None:
#             self.log_min_variance = None
#         else:
#             self.log_min_variance = float(np.log(min_variance))
#         self.K = K
#         self.input_channels = input_channels
#         self.imsize = imsize
#         self.imlength = self.imsize * self.imsize * self.input_channels
#         self.refinement_net = refinement_net
#         self.decoder_imsize = decoder.input_width
#         self.beta = beta
#         self.physics_net = physics_net
#         self.lstm_size = 256
#         # self.train_T = train_T
#         self.test_T = test_T
#         self.seed_steps = seed_steps
#         self.schedule_type = schedule_type
#
#         self.decoder = decoder
#
#         # if action_dim is not None:
#         #     self.action_encoder = Mlp((128,), 32, action_dim,
#         #                                  hidden_activation=nn.ELU())
#         #     self.action_lambda_encoder = Mlp((256, 256), representation_size, representation_size+32,
#         #                                  hidden_activation=nn.ELU())
#
#         l_norm_sizes = [7, 1, 1]
#         self.layer_norms = nn.ModuleList([LayerNorm2D(l) for l in l_norm_sizes])
#
#         self.epoch = 0
#
#         self.apply(ptu.init_weights)
#         self.lambdas1 = Parameter(ptu.zeros((self.representation_size)))
#         self.lambdas2 = Parameter(ptu.ones((self.representation_size)) * 0.6)
#
#         self.sigma = from_numpy(np.array([sigma]))
#
#         self.eval_mode = False
#
#     def encode(self, input):
#         pass
#
#     def set_eval_mode(self, eval):
#         self.eval_mode = eval
#
#     def decode(self, lambdas1, lambdas2, inputK, bs):
#         # RV: inputK: (bs*K, ch, imsize, imsize)
#         # RV: lambdas1, lambdas2: (bs*K, lstm_size)
#
#         latents = self.rsample_softplus([lambdas1, lambdas2])  # lambdas1, lambdas2 are mu, softplus
#
#         broadcast_ones = ptu.ones((latents.shape[0], latents.shape[1], self.decoder_imsize, self.decoder_imsize)).to(
#             latents.device)  # RV: (bs, lstm_size, decoder_imsize. decoder_imsize)
#         decoded = self.decoder(latents, broadcast_ones)  # RV: Uses broadcast decoding network, output (bs*K, 4, D, D)
#         # print("decoded.shape: {}".format(decoded.shape))
#         x_hat = decoded[:, :3]  # RV: (bs*K, 3, D, D)
#         m_hat_logits = decoded[:, 3]  # RV: (bs*K, 1, D, D)
#
#         m_hat_logit = m_hat_logits.view(bs, self.K, self.imsize, self.imsize)  # RV: (bs, K, D, D)
#         mask = F.softmax(m_hat_logit, dim=1)  # (bs, K, D, D), mask probabilities
#
#         if inputK is not None:
#             pixel_x_prob = self.gaussian_prob(x_hat, inputK, self.sigma).view(bs, self.K, self.imsize,
#                                                                               self.imsize)  # RV: Component p(x|h), (bs,K,D,D)
#             pixel_likelihood = (mask * pixel_x_prob).sum(
#                 1)  # sum along K  #RV:sum over k of m_k*p_k, complete log likelihood
#             log_likelihood = -torch.log(pixel_likelihood + 1e-12).sum() / bs  # RV: This should be complete log likihood?
#
#             kle = self.kl_divergence_softplus([lambdas1, lambdas2])
#             kle_loss = self.beta * kle.sum() / bs  # RV: KL loss
#             loss = log_likelihood + kle_loss  # RV: Total loss
#         else:
#             pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood = None, None, None, None, None
#
#         return x_hat, mask, m_hat_logits, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood
#
#     def gaussian_prob(self, inputs, targets, sigma):
#         ch = 3
#         # (2pi) ^ ch = 248.05
#         sigma = sigma.to(inputs.device)
#         return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (
#                 torch.sqrt(sigma ** (2 * ch)) * 248.05)
#
#     def logprob(self, inputs, obs_distribution_params):
#         pass
#
#     def gaussian_log_prob(self, inputs, targets, sigma):
#         return torch.pow(inputs - targets, 2) / (2 * sigma ** 2)
#
#     def forward(self, input, actions=None, schedule=None, seedsteps=5):
#
#         return self._forward_dynamic_actions(input, actions, schedule)
#
#     def initialize_hidden(self, bs):
#         return (ptu.from_numpy(np.zeros((bs, self.lstm_size))),
#                 ptu.from_numpy(np.zeros((bs, self.lstm_size))))
#
#     def plot_latents(self, ground_truth, masks, x_hats, mse, idx):
#         K = self.K
#         imsize = self.imsize
#         T = masks.shape[1]
#         m = masks[idx].permute(1, 0, 2, 3, 4).repeat(1, 1, 3, 1, 1)  # (K, T, ch, imsize, imsize)
#         x = x_hats[idx].permute(1, 0, 2, 3, 4)
#         rec = (m * x)
#         full_rec = rec.sum(0, keepdim=True)
#
#         comparison = torch.cat([ground_truth, full_rec, m, rec], 0).view(-1, 3, imsize, imsize)
#         save_image(comparison.data.cpu(), logger.get_snapshot_dir() + '/goal_latents_%0.5f.png' % mse, nrow=T)
#
#     def refine(self, input, hidden_state, plot_latents=False):
#         K = self.K
#         bs = 8
#         input = input.repeat(bs, 1, 1, 1).unsqueeze(1)
#
#         T = 7  # Refine for 7 steps
#
#         outputs = [[], [], [], [], []]
#         # Run multiple times to get best one
#         for i in range(6):
#             x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = self._forward_dynamic_actions(
#                 input, None,
#                 schedule=np.zeros((T)))
#             outputs[0].append(x_hats)
#             outputs[1].append(masks)
#             outputs[2].append(final_recon)
#             outputs[3].append(lambdas[0].view(-1, K, self.representation_size))
#
#         x_hats = torch.cat(outputs[0], 0)
#         masks = torch.cat(outputs[1], 0)
#         final_recon = torch.cat(outputs[2])
#         lambdas = torch.cat(outputs[3], 0)
#
#         lambda_recon = (x_hats * masks)
#         recon = torch.clamp(final_recon, 0, 1)
#         mse = torch.pow(final_recon - input[0], 2).mean(3).mean(2).mean(1)
#         best_idx = torch.argmin(mse)
#         if plot_latents:
#             mses, best_idxs = mse.sort()
#             for i in range(8):
#                 self.plot_latents(input[0].unsqueeze(0).repeat(1, T, 1, 1, 1), masks,
#                                   x_hats, mse[best_idxs[i]], best_idxs[i])
#
#         best_lambda = lambdas[best_idx]
#
#         return recon[best_idx].data, best_lambda.data, lambda_recon[best_idx, -1].data, masks[best_idx,
#                                                                                               -1].data.squeeze()
#
#     # input:tensor of shape (B, 3, D, D) or (B, T1, 3, D, D)
#     # actions: (B, T2, 3, D, D)
#     def step(self, input, actions, initial_lambdas, schedule=None, plot_latents=False):
#         if len(input.shape) == 4:
#             input = input.unsqueeze(1)
#
#         K = self.K
#         bs = input.shape[0]
#
#         # schedule = create_schedule(False, self.test_T, self.schedule_type, self.seed_steps) #RV: Returns schedule of 1's and 0's
#         if schedule is None:
#             T1 = input.shape[1]
#             seed_steps = self.seed_steps
#             initial_len = seed_steps + (T1-1) * 2 #Refine for self.seed_steps on first image, then go physics, refine for every next step
#             if actions is not None:
#                 rollout_len = actions.shape[1] - (T1 - 1)
#             else:
#                 rollout_len = 0
#             schedule = np.zeros((initial_len + rollout_len))
#             schedule[seed_steps:initial_len:2] = 1 #Do a physics and refine step for every input T1
#             schedule[initial_len:] = 1 #Do physics steps for the rest
#
#         x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = self._forward_dynamic_actions(
#             input, actions, schedule=schedule)
#
#         lambda_recon = (x_hats * masks)
#         recon = torch.clamp(final_recon, 0, 1)
#         if plot_latents:
#             # i = 0
#             # self.plot_latents_trunc(input[0].unsqueeze(0).repeat(1, len(masks), 1, 1, 1), masks,
#             #                         x_hats, 0)
#
#             imsize = 64
#             m = masks[0].permute(1, 0, 2, 3, 4).repeat(1, 1, 3, 1, 1)  # (K, T, ch, imsize, imsize)
#             x = x_hats[0].permute(1, 0, 2, 3, 4)
#             rec = (m * x)
#             full_rec = rec.sum(0, keepdim=True)
#
#             comparison = torch.cat([input[0, :self.test_T].unsqueeze(0), full_rec, m, rec],
#                                    0).view(-1, 3, imsize, imsize)
#             # import pdb; pdb.set_trace()
#
#             if isinstance(plot_latents, str):
#                 name = logger.get_snapshot_dir() + plot_latents
#             else:
#                 name = logger.get_snapshot_dir() + '/test.png'
#
#             save_image(comparison.data.cpu(), name, nrow=self.test_T)
#             # save_image(comparison.data.cpu(), logger.get_snapshot_dir() + '/test.png', nrow=self.test_T)
#         #  x_hats, 0, i)
#         # pred_obs, obs_latents, obs_latents_recon
#
#         return recon.data, lambdas[0].view(bs, K, -1).data, lambda_recon[:, -1].data
#
#     # Batch method for step
#     def step_batched(self, inputs, actions, bs=4):
#         # Handle large obs in batches
#         n_batches = int(np.ceil(inputs.shape[0] / float(bs)))
#         outputs = [[], [], []]
#
#         for i in range(n_batches):
#             start_idx = i * bs
#             end_idx = min(start_idx + bs, inputs.shape[0])
#             if actions is not None:
#                 actions_batch = actions[start_idx:end_idx]
#             else:
#                 actions_batch = None
#
#             pred_obs, obs_latents, obs_latents_recon = self.step(inputs[start_idx:end_idx], actions_batch)
#             outputs[0].append(pred_obs)
#             outputs[1].append(obs_latents)
#             outputs[2].append(obs_latents_recon)
#
#         return torch.cat(outputs[0]), torch.cat(outputs[1]), torch.cat(outputs[2])
#
#     # RV: Inputs: Information needed for IODINE refinement network (note much more information needed than RNEM)
#     # RV: Outputs: Updates lambdas and hs
#     def refine_lambdas(self, pixel_x_prob, pixel_likelihood, mask, m_hat_logit, loss, x_hat,
#                        lambdas1, lambdas2, inputK, latents, h1, h2, tiled_k_shape, bs):
#         K = self.K
#         lns = self.layer_norms
#         posterior_mask = pixel_x_prob / (pixel_x_prob.sum(1, keepdim=True) + 1e-8)  # avoid divide by zero
#         leave_out_ll = pixel_likelihood.unsqueeze(1) - mask * pixel_x_prob
#         x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = \
#             torch.autograd.grad(loss, [x_hat, mask] + [lambdas1, lambdas2], create_graph=not self.eval_mode,
#                                 retain_graph=not self.eval_mode)
#
#         a = torch.cat([
#             torch.cat([inputK, x_hat, mask.view(tiled_k_shape), m_hat_logit.view(tiled_k_shape)], 1),
#             lns[0](torch.cat([
#                 x_hat_grad.detach(),
#                 mask_grad.view(tiled_k_shape).detach(),
#                 posterior_mask.view(tiled_k_shape).detach(),
#                 pixel_likelihood.unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape).detach(),
#                 leave_out_ll.view(tiled_k_shape).detach()], 1))
#
#         ], 1)
#
#         extra_input = torch.cat([lns[1](lambdas_grad_1.view(bs * K, -1).detach()),
#                                  lns[2](lambdas_grad_2.view(bs * K, -1).detach())
#                                  ], -1)
#
#         lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2,
#                                                          extra_input=torch.cat(
#                                                              [extra_input, lambdas1, lambdas2, latents], -1))
#         return lambdas1, lambdas2, h1, h2
#
#     # RV: Input is (bs, T1, ch, imsize, imsize), schedule is (T2,): 0 for refinement and 1 for physics
#     #    Runs refinement/dynamics on input accordingly into
#     def _forward_dynamic_actions(self, input, actions, schedule):
#         K = self.K
#         bs = input.shape[0]
#         T2 = schedule.shape[0]
#
#         # means and log_vars of latent
#         lambdas1 = self.lambdas1.unsqueeze(0).repeat(bs * K, 1)
#         lambdas2 = self.lambdas2.unsqueeze(0).repeat(bs * K, 1)
#         # initialize hidden state
#         h1, h2 = self.initialize_hidden(bs * K)  # RV: Each one is (bs, self.lstm_size)
#
#         h1 = h1.to(input.device)
#         h2 = h2.to(input.device)
#
#         losses, x_hats, masks = [], [], []
#         untiled_k_shape = (bs, K, -1, self.imsize, self.imsize)
#         tiled_k_shape = (bs * K, -1, self.imsize, self.imsize)
#
#         current_step = 0
#
#         inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)  # RV: (bs*K, ch, imsize, imsize)
#         x_hat, mask, m_hat_logit, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood = self.decode(
#             lambdas1, lambdas2, inputK, bs)  # RV: Returns sampled latents, decoded outputs, and computes the likelihood/loss
#         losses.append(loss)
#
#         total_loss_weight = 0
#         for t in range(1, T2 + 1):
#             if lambdas1.shape[0] % self.K != 0:
#                 print("UH OH: {}".format(t))
#             # Refine
#             if schedule[t - 1] == 0:
#                 inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)  # RV: (bs*K, ch, imsize, imsize)
#                 lambdas1, lambdas2, h1, h2 = self.refine_lambdas(pixel_x_prob, pixel_likelihood, mask, m_hat_logit,
#                                                                  loss, x_hat, lambdas1, lambdas2, inputK, latents, h1,
#                                                                  h2, tiled_k_shape, bs)  # RV: Update lambdas and h's using info
#                 # if not applied_action: # Do physics on static scene if haven't applied action yet
#             # Physics
#             else:
#                 current_step += 1
#                 if actions is not None:
#                     actionsK = actions[:, current_step - 1].unsqueeze(1).repeat(1, K, 1).view(bs * K, -1)
#                 else:
#                     actionsK = None
#
#                 # inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)
#                 lambdas1, _ = self.physics_net(lambdas1, lambdas2, actionsK)
#
#             loss_w = get_loss_weight(t, schedule, self.schedule_type)
#
#             # Decode and get loss
#             x_hat, mask, m_hat_logit, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood = \
#                 self.decode(lambdas1, lambdas2, inputK, bs)
#
#             x_hats.append(x_hat.data)
#             masks.append(mask.data)
#             if inputK is not None:
#                 total_loss_weight += loss_w
#                 losses.append(loss * loss_w)
#
#         total_loss = sum(losses) / total_loss_weight
#
#         final_recon = (mask.unsqueeze(2) * x_hat.view(untiled_k_shape)).sum(1)
#         mse = torch.pow(final_recon - input[:, -1], 2).mean()
#
#         all_x_hats = torch.stack([x.view(untiled_k_shape) for x in x_hats], 1)  # (bs, T, K, 3, imsize, imsize)
#         all_masks = torch.stack([x.view(untiled_k_shape) for x in masks], 1)  # # (bs, T, K, 1, imsize, imsize)
#         return all_x_hats.data, all_masks.data, total_loss, kle_loss.data / self.beta, \
#                log_likelihood.data, mse, final_recon.data, [lambdas1.data, lambdas2.data]





#########New cleaned version#########
#Variant must contain the following keywords: refinement_model_type, decoder_model_type,  dynamics_special_model_type,
#    dynamics_model_type
def create_model_v2(variant, repsize, action_dim):
    K = variant['K']

    ref_model_args = Refinement_Args[variant["refinement_model_type"]]
    if ref_model_args[0] == "reg":
        refinement_kwargs = ref_model_args[1](repsize)
        refinement_net = RefinementNetwork_v2(**refinement_kwargs)
    elif ref_model_args[0] == "sequence_iodine":
        refinement_kwargs = ref_model_args[1](repsize, action_dim)
        refinement_net = RefinementNetwork_v2(**refinement_kwargs)
    else:
        raise ValueError("{}".format(ref_model_args[0]))

    dec_model_args = Decoder_Args[variant["decoder_model_type"]]
    if dec_model_args[0] == "reg":
        decoder_kwargs = dec_model_args[1](repsize)
        decoder_net = DecoderNetwork_V2(**decoder_kwargs)
    else:
        raise ValueError("{}".format(dec_model_args[0]))

    dyn_model_args = Physics_Args[variant["dynamics_model_type"]]
    if dyn_model_args[0] == "reg":
        physics_kwargs = dyn_model_args[1](repsize, action_dim)
        dynamics_net = PhysicsNetwork_v2(**physics_kwargs)
    elif dyn_model_args[0] == "mlp":
        physics_kwargs = dyn_model_args[1](repsize, action_dim)
        dynamics_net = PhysicsNetworkMLP_v2(K, **physics_kwargs)
    else:
        raise ValueError("{}".format(variant["dynamics_model_type"][0]))


    model = IodineVAE_v2(refinement_net, dynamics_net, decoder_net, repsize)
    model.set_k(K)
    return model


class IodineVAE_v2(torch.nn.Module):
    def __init__(self, refine_net, dynamics_net, decode_net, repsize):
        super().__init__()
        self.refinement_net = refine_net
        self.dynamics_net = dynamics_net
        self.decode_net = decode_net
        self.K = None
        self.repsize = repsize

        #Loss hyper-parameters
        self.sigma = 0.1 #ptu.from_numpy(np.array([0.1]))
        self.beta = 5

        #Refinement variables
        l_norm_sizes = [7, 1, 1]
        self.layer_norms = nn.ModuleList([LayerNorm2D(l) for l in l_norm_sizes])
        self.eval_mode = False
        self.lambdas1 = Parameter(ptu.zeros((self.repsize)))
        self.lambdas2 = Parameter(ptu.ones((self.repsize)) * 0.6)

    def set_k(self, k):
        self.K = k
        self.dynamics_net.set_k(k)

    #Input: x: (a,b,c,d,...)
    #Output: y: (a*b,c,d,...)
    def _flatten_first_two(self, x):
        return x.view([x.shape[0]*x.shape[1]] + list(x.shape[2:]))

    #Input: x: (bs*k,a,b,...)
    #Output: y: (bs,k,a,b,....)
    def _unflatten_first(self, x, k):
        return x.view([-1, k] + list(x.shape[1:]))

    def _gaussian_prob(self, inputs, targets, sigma):
        ch = 3
        # (2pi) ^ ch = 248.05
        # sigma = sigma.to(inputs.device)
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (np.sqrt(sigma ** (2 * ch)) * 248.05)

    #Inputs: latent_distribution_params: [mu, softplus], each (B,R)
    #Outputs: A random sample of the same size based off mu and softplus (B,R)
    def rsample_softplus(self, latent_distribution_params):
        mu, softplus = latent_distribution_params
        stds = torch.sqrt(torch.log(1 + softplus.exp()))

        # stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size()).to(stds.device)  # RV: Is the star necessary?
        latents = epsilon * stds + mu
        return latents

    def kl_divergence_softplus(self, latent_distribution_params):
        mu, softplus = latent_distribution_params
        stds = torch.sqrt(torch.log(1 + softplus.exp()))
        logvar = torch.log(torch.log(1 + softplus.exp()))
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - stds, dim=1).mean()


    #Input: Integer n denoting how many hidden states to initialize
    #Output: Returns hidden states, tuples of (n,k,repsize) and (n,k,lstm_size)
    def _get_initial_hidden_states(self, n):
        k = self.K
        lambdas1 = self._unflatten_first(self.lambdas1.unsqueeze(0).repeat(n*k, 1), k) #(n,k,repsize)
        lambdas2 = self._unflatten_first(self.lambdas2.unsqueeze(0).repeat(n*k, 1), k) #(n,k,repsize)
        h1, h2 = self.refinement_net.initialize_hidden(n*k) #Each (n*k,lstm_size)
        h1, h2 = self._unflatten_first(h1, k), self._unflatten_first(h2, k) #Each (n,k,lstm_size)
        return [lambdas1, lambdas2, h1, h2]

    #Inputs: images: (B, 3, D, D),  hidden_states: Tuples of (B, K, R),  action: None or (B, A)
    #Outputs: new_hidden_states: Tuples of (B, K, R)
    def refine(self, hidden_states, images, action=None):
        bs, imsize = images.shape[0], images.shape[2]
        K = self.K
        tiled_k_shape = (bs*K, -1, imsize, imsize)

        lambdas1, lambdas2 = self._flatten_first_two(hidden_states[0]), self._flatten_first_two(hidden_states[1])  # Each (B*K, R)
        colors, mask, mask_logits, sampled_latents, color_probs, pixel_complete_log_likelihood, kle_loss, loss, complete_log_likelihood\
            = self.decode(hidden_states, images)

        posterior_mask = color_probs / (color_probs.sum(1, keepdim=True) + 1e-8)  #(B,K,D,D)
        leave_out_ll = pixel_complete_log_likelihood.unsqueeze(1) - mask.squeeze(2) * color_probs #(B,K,D,D)

        pdb.set_trace()
        x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = \
            torch.autograd.grad(loss, [colors, mask.squeeze(2), lambdas1, lambdas2], create_graph=not self.eval_mode,
                                retain_graph=not self.eval_mode)

        k_images = images.unsqueeze(1).repeat(1, K, 1, 1, 1) #(B,K,3,D,D)

        lns = self.layer_norms
        a = torch.cat([
            torch.cat([k_images.view(tiled_k_shape), colors.view(tiled_k_shape), mask.view(tiled_k_shape), mask_logits.view(tiled_k_shape)], 1),
            lns[0](torch.cat([
                x_hat_grad.detach(),
                mask_grad.view(tiled_k_shape).detach(),
                posterior_mask.view(tiled_k_shape).detach(),
                pixel_complete_log_likelihood.unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape).detach(),
                leave_out_ll.view(tiled_k_shape).detach()], 1))
        ], 1)

        extra_input = torch.cat([lns[1](lambdas_grad_1.view(bs * K, -1).detach()),
                                 lns[2](lambdas_grad_2.view(bs * K, -1).detach())
                                 ], -1)

        if action is not None: #Use action as extra input into refinement: This is only for next step refinement (sequence iodine)
            action = action.unsqueeze(1).repeat(1,K,1) #(B*K,A)

        h1, h2 = self._flatten_first_two(hidden_states[2]), self._flatten_first_two(hidden_states[3]) #(B*K, R2)
        lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2,
                                                         extra_input=torch.cat(
                                                             [extra_input, lambdas1, lambdas2, self._flatten_first_two(sampled_latents)], -1),
                                                         add_fc_input=action) #Lambdas (B*K,R),   h (B*K,R2)

        new_hidden_states = [self._unflatten_first(lambdas1, K), self._unflatten_first(lambdas2, K), self._unflatten_first(h1, K), self._unflatten_first(h2, K)]
        return new_hidden_states

    #Inputs: actions: (B, A),  hidden_states: Tuples of (B, K, R)
    #Outputs: new_hidden_states: Tuples of (B, K, R)
    def dynamics(self, hidden_states, actions):
        bs, k = hidden_states[0].shape[:2] #hidden_states[0] is lambdas1 (B,K,R)
        lambdas1, lambdas2 = self._flatten_first_two(hidden_states[0]), self._flatten_first_two(hidden_states[1]) #Each (B*K,R)
        actions = actions.unsqueeze(1).repeat(1, k, 1) #(B,K,A)
        actions = self._flatten_first_two(actions) #(B*K,A)
        lambdas1, lambdas2 = self.dynamics_net(lambdas1, lambdas2, actions) #(B*K,R)
        lambdas1 = self._unflatten_first(lambdas1, k) #(B,K,R)
        lambdas2 = self._unflatten_first(lambdas2, k) #(B,K,R)

        h1, h2 = ptu.zeros_like(hidden_states[2]), ptu.zeros_like(hidden_states[3]) #Set the h's to zero as the next refinement should start from scratch (B,K,R2)
        return [lambdas1, lambdas2, h1, h2]

    #Inputs: hidden_states: Tuples of (B, K, R), target_imgs: None or (B, 3, D, D)
    #Outputs:
    #  If target_imgs is None: colors (B,K,3,D,D), masks (B,K,1,D,D)
    #  Else, compute the relevant losses as well:
    #  colors (B,K,3,D,D), masks (B,K,1,D,D), mask_logits (B,K,1,D,D), sampled_latents (B,K,R), color_probs (B,K,D,D)
    #  pixel_complete_log_likelihood (B,D,D), kle_loss (Sc), loss (Sc), complete_log_likelihood (Sc)
    #    Note: The outputted losses are normalized by batch size
    def decode(self, hidden_states, target_imgs=None):
        bs, k = hidden_states[0].shape[:2] #hidden_states[0] is lambdas1 (B,K,R)
        lambdas1, lambdas2 = self._flatten_first_two(hidden_states[0]), self._flatten_first_two(hidden_states[1]) #Each (B*K, R)

        sampled_latents = self.rsample_softplus([lambdas1, lambdas2]) #lambdas1, lambdas2 are mu, softplus;  sampled_latents: (B*K,R)
        mask_logits, colors = self.decode_net(sampled_latents) #mask_logits: (B*K,1,D,D),  colors: (B*K,3,D,D)

        mask_logits = self._unflatten_first(mask_logits, k) #(B,K,1,D,D)
        mask = F.softmax(mask_logits, dim=1)  #(B,K,1,D,D), these are the mask probability values
        colors = self._unflatten_first(colors, k) #(B,K,3,D,D)
        # final_recon = (mask * colors).sum(1) #(B,3,D,D)

        if target_imgs is not None:
            k_targs = target_imgs.unsqueeze(1).repeat(1, k, 1, 1, 1) #(B,3,D,D) -> (B,1,3,D,D) -> (B,K,3,D,D)
            k_targs = self._flatten_first_two(k_targs) #(B,K,3,D,D) -> (B*K,3,D,D)
            tmp_colors = self._flatten_first_two(colors) #(B,K,3,D,D) -> (B*K,3,D,D)
            color_probs = self._gaussian_prob(tmp_colors, k_targs, self.sigma) #Computing p(x|h),  (B*K,D,D)
            color_probs = self._unflatten_first(color_probs, k) #(B,K,D,D)
            pixel_complete_log_likelihood = (mask.squeeze(2)*color_probs).sum(1) #Sum over K, pixelwise complete log likelihood (B,D,D)
            complete_log_likelihood = -torch.log(pixel_complete_log_likelihood + 1e-12).sum()/bs #(Scalar)

            kle = self.kl_divergence_softplus([lambdas1, lambdas2])
            kle_loss = self.beta * kle.sum() / bs #KL loss, (Sc)

            total_loss = complete_log_likelihood + kle_loss #Total loss, (Sc)
            sampled_latents = self._unflatten_first(sampled_latents, k) #(B,K,R)
            return colors, mask, mask_logits, sampled_latents, color_probs, pixel_complete_log_likelihood, kle_loss, total_loss, complete_log_likelihood
        else:
            return colors, mask


    #Inputs: images: (B, T_obs, 3, D, D),  actions: (B, T_acs, A),  initial_hidden_state: Tuples of (B, K, repsize)
    #   schedule: (T1),   loss_schedule:(T1)
    #Output: colors_list (T1,B,K,3,D,D), masks_list (T1,B,K,1,D,D), final_recon (B,3,D,D),
    # total_loss, total_kle_loss, total_clog_prob, mse are all (Sc)
    def run_schedule(self, images, actions, initial_hidden_state, schedule, loss_schedule):
        b = images.shape[0]
        if initial_hidden_state is None: #Initialize initial_hidden_state if it is not passed in
            initial_hidden_state = self._get_initial_hidden_states(b)

        #Save outputs: colors_list (T1,B,K,3,D,D),  masks (T1,B,K,1,D,D),  losses_list (T1)
        colors_list, masks_list, losses_list, kle_loss_list, clog_prob_list = [], [], [], [], []

        current_step = 0
        cur_hidden_state = initial_hidden_state
        for i in range(len(schedule)):
            if schedule[i] == 0: #Refinement step
                input_img = images[:, current_step] #(B,3,D,D)
                cur_hidden_state = self.refine(cur_hidden_state, input_img)
            elif schedule[i] == 1: #Physics step
                if actions is not None:
                    input_actions = actions[:, current_step] #(B,A)
                else:
                    input_actions = None
                cur_hidden_state = self.dynamics(cur_hidden_state, input_actions)
                current_step += 1
            elif schedule[i] == 2: #Next step refinement, just for sequence iodine
                if actions is not None:
                    input_actions = actions[:, current_step] #(B,A)
                else:
                    input_actions = None
                input_img = images[:, current_step]
                cur_hidden_state = self.refine(cur_hidden_state, input_img, action=input_actions)
                current_step += 1
            else:
                raise ValueError("Invalid schedule entry: {}".format(schedule[i]))

            if loss_schedule[i] != 0:
                target_images = images[:, current_step] #(B,3,D,D)
                colors, mask, mask_logits, sampled_latents, color_probs, pixel_clog_prob, kle_loss, total_loss, clog_prob \
                    = self.decode(cur_hidden_state, target_imgs=target_images)

                colors_list.append(colors)
                masks_list.append(mask)

                losses_list.append(total_loss * loss_schedule[i])
                kle_loss_list.append(total_loss * kle_loss)
                clog_prob_list.append(total_loss * clog_prob)
            else:
                colors, mask = self.decode(cur_hidden_state, target_imgs=None)
                colors_list.append(colors)
                masks_list.append(mask)

        colors_list = torch.stack(colors_list) #(T1,B,K,3,D,D)
        masks_list = torch.stack(masks_list) #(T1,B,K,1,D,D)

        if sum(loss_schedule) == 0:
            total_loss, total_kle_loss, total_clog_prob = None, None, None
        else:
            sum_loss_weights = sum(loss_schedule)
            total_loss = sum(losses_list) / sum_loss_weights #Scalar
            total_kle_loss = sum(kle_loss_list) / sum_loss_weights
            total_clog_prob = sum(clog_prob_list) / sum_loss_weights

        final_recon = (colors_list[-1] * masks_list[-1]).sum(1) #(B,K,3,D,D) -> (B,3,D,D)
        mse = torch.pow(final_recon - images[:, -1], 2).mean() #(B,3,D,D) -> (Sc)

        return colors_list, masks_list, final_recon, total_loss, total_kle_loss, total_clog_prob, mse

    #Wrapper for self.run_schedule() required for training with DataParallel
    def forward(self, images, actions, initial_hidden_state, schedule, loss_schedule):
        return self.run_schedule(images, actions, initial_hidden_state, schedule, loss_schedule)
