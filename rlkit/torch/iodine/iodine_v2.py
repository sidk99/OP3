import torch
import torch.utils.data
from rlkit.torch.iodine.physics_network_v2 import PhysicsNetwork_v2, PhysicsNetworkMLP_v2, Physics_Args
from rlkit.torch.iodine.refinement_network_v2 import RefinementNetwork_v2, Refinement_Args
from rlkit.torch.iodine.decoder_network_v2 import DecoderNetwork_V2, Decoder_Args
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

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


# model = load_from_old_version(model, "/nfs/kun1/users/rishiv/Research/op3_exps/08-05-twoBalls-static-iodine-reg/08-05-twoBalls-static_iodine-reg_2019_08_05_14_19_42_0000--s-747/test_save_params.pkl")
# def load_from_old_version(model, file_path):
#     cur_model_state_dict = model.state_dict()
#     vals = torch.load(file_path)
#     for a_key in vals.keys():
#         if "decoder" in a_key:
#             cur_model_key = a_key.replace("module.decoder", "decode_net.broadcast_net")
#             cur_model_state_dict[cur_model_key] = vals[a_key]
#         elif "refinement" in a_key:
#             cur_model_key = a_key.replace("module.", "")
#             cur_model_state_dict[cur_model_key] = vals[a_key]
#         elif "layer_norms" in a_key:
#             cur_model_key = a_key.replace("module.", "")
#             cur_model_state_dict[cur_model_key] = vals[a_key]
#
#     model.load_state_dict(cur_model_state_dict)
#     return model


#Variant must contain the following keywords: refinement_model_type, decoder_model_type, dynamics_model_type, K
#Note: repsize represents the size of the deterministic & stochastic each, so the full state size is repsize*2
def create_model_v2(variant, det_size, sto_size, action_dim):
    K = variant['K']

    ref_model_args = Refinement_Args[variant["refinement_model_type"]]
    if ref_model_args[0] == "reg":
        refinement_kwargs = ref_model_args[1](sto_size)
        refinement_net = RefinementNetwork_v2(**refinement_kwargs)
    elif ref_model_args[0] == "sequence_iodine":
        refinement_kwargs = ref_model_args[1](sto_size, action_dim)
        refinement_net = RefinementNetwork_v2(**refinement_kwargs)
    else:
        raise ValueError("{}".format(ref_model_args[0]))

    dec_model_args = Decoder_Args[variant["decoder_model_type"]]
    if dec_model_args[0] == "reg":
        decoder_kwargs = dec_model_args[1](det_size + sto_size)
        decoder_net = DecoderNetwork_V2(**decoder_kwargs)
    else:
        raise ValueError("{}".format(dec_model_args[0]))

    dyn_model_args = Physics_Args[variant["dynamics_model_type"]]
    if dyn_model_args[0] == "reg":
        physics_kwargs = dyn_model_args[1](det_size, sto_size, action_dim)
        dynamics_net = PhysicsNetwork_v2(**physics_kwargs)
    elif dyn_model_args[0] == "mlp":
        physics_kwargs = dyn_model_args[1](det_size, sto_size, action_dim)
        dynamics_net = PhysicsNetworkMLP_v2(K, **physics_kwargs)
    else:
        raise ValueError("{}".format(variant["dynamics_model_type"][0]))

    model = IodineVAE_v2(refinement_net, dynamics_net, decoder_net, det_size, sto_size)
    model.set_k(K)
    return model


#Probably should rename to OP3
#Notation: R denotes size of deterministic & stochastic state (each), R2 denotes dize of lstm hidden state for iodine
#  D denotes image size, A denotes action dimension, B denotes batch size
#  (Sc) denotes a scalar while (X,Y,Z) represent the shape
class IodineVAE_v2(torch.nn.Module):
    def __init__(self, refine_net, dynamics_net, decode_net, det_size, sto_size):
        super().__init__()
        self.refinement_net = refine_net
        self.dynamics_net = dynamics_net
        self.decode_net = decode_net
        self.K = None
        self.det_size = det_size
        self.sto_size = sto_size

        #Loss hyper-parameters
        self.sigma = 0.1 #ptu.from_numpy(np.array([0.1]))
        self.beta = 1

        #Refinement variables
        l_norm_sizes = [7, 1, 1]
        self.layer_norms = nn.ModuleList([LayerNorm2D(l) for l in l_norm_sizes])
        self.eval_mode = False

        #Initial state parameters
        self.inital_deter_state = Parameter(ptu.zeros((self.det_size)))
        self.initial_lambdas1 = Parameter(ptu.zeros((self.sto_size)))
        self.initial_lambdas2 = Parameter(ptu.ones((self.sto_size)) * 0.6)

    def set_k(self, k):
        self.K = k
        self.dynamics_net.set_k(k)

    #Input: x: (a,b,*)
    #Output: y: (a*b,*)
    def _flatten_first_two(self, x):
        if x is None:
            return x
        return x.view([x.shape[0]*x.shape[1]] + list(x.shape[2:]))

    #Input: x: (bs*k,*)
    #Output: y: (bs,k,*)
    def _unflatten_first(self, x, k):
        if x is None:
            return x
        return x.view([-1, k] + list(x.shape[1:]))

    #Input: inputs (B*K,3,D,D), targets (B*K,3,D,D), sigma (Sc)
    #Output: (B*K,D,D)
    def _gaussian_prob(self, inputs, targets, sigma):
        ch = 3
        # (2pi) ^ ch = 248.05
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (np.sqrt(sigma ** (2 * ch)) * 248.05)

    # def gaussian_prob(self, inputs, targets, sigma):
    #     ch = 3
    #     # (2pi) ^ ch = 248.05
    #     sigma = sigma.to(inputs.device)
    #     return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (
    #             torch.sqrt(sigma ** (2 * ch)) * 248.05)

    #Inputs: latent_distribution_params: [mu, softplus], each (B,R)
    #Outputs: A random sample of the same size based off mu and softplus (B,R)
    # def rsample_softplus(self, latent_distribution_params):
    #     mu, softplus = latent_distribution_params
    #     stds = torch.sqrt(torch.log(1 + softplus.exp()))
    #
    #     # stds = (0.5 * logvar).exp()
    #     epsilon = ptu.randn(*mu.size()).to(stds.device)  # RV: Is the star necessary?
    #     latents = epsilon * stds + mu
    #     return latents

    # def kl_divergence_softplus(self, latent_distribution_params):
    #     mu, softplus = latent_distribution_params
    #     stds = torch.sqrt(torch.log(1 + softplus.exp()))
    #     logvar = torch.log(torch.log(1 + softplus.exp()))
    #     return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - stds.pow(2), dim=1).sum() #Note: Incorrect!


    #Compute the kl loss between prior and posterior
    #Note: This is NOT normalized
    def kl_divergence_prior_post(self, prior, post):
        mu1, softplus1 = prior["lambdas1"], prior["lambdas2"]
        mu2, softplus2 = post["lambdas1"], post["lambdas2"]

        stds1 = torch.sqrt(torch.log(1 + softplus1.exp()))
        stds2 = torch.sqrt(torch.log(1 + softplus2.exp()))
        q1 = MultivariateNormal(loc=mu1, scale_tril=torch.diag_embed(stds1))
        q2 = MultivariateNormal(loc=mu2, scale_tril=torch.diag_embed(stds2))
        return torch.distributions.kl.kl_divergence(q2, q1) #KL(post||prior), note ordering matters!

    def get_samples(self, means, softplusses):
        stds = torch.sqrt(torch.log(1 + softplusses.exp()))
        epsilon = ptu.randn(*means.size()).to(stds.device)
        latents = epsilon * stds + means
        return latents

    def get_full_tensor_state(self, hidden_state):
        if self.det_size == 0:
            return hidden_state["post"]["samples"]
        return torch.cat([hidden_state["post"]["deter_state"], hidden_state["post"]["samples"]], dim=2) #(B,K,R)

    #Input: Integer n denoting how many hidden states to initialize
    #Output: Returns hidden states, tuples of (n*k,repsize) and (n*k,lstm_size)
    def _get_initial_hidden_states(self, n, device):
        k = self.K

        if self.det_size == 0:
            deter_state = None
        else:
            deter_state = self._unflatten_first(self.inital_deter_state.unsqueeze(0).repeat(n*k, 1), k) #(n,k,Rd)

        lambdas1 = self._unflatten_first(self.initial_lambdas1.unsqueeze(0).repeat(n*k, 1), k) #(n,k,Rs)
        lambdas2 = self._unflatten_first(self.initial_lambdas2.unsqueeze(0).repeat(n*k, 1), k) #(n,k,Rs)
        samples = self.get_samples(lambdas1, lambdas2) #(n,k,R=Rd+Rs)
        h1, h2 = self.refinement_net.initialize_hidden(n * k)  # Each (1,n*k,lstm_size)
        h1, h2 = h1.to(device), h2.to(device)

        hidden_state = {
            "prior" : {
                "lambdas1": torch.zeros((n, k, self.sto_size)).to(device),  # (n*k,Rs)
                "lambdas2": torch.log(torch.exp(torch.ones((n, k, self.sto_size))) - 1).to(device), #log(e-1) as softplus (n*k,Rs)
            },
            "post" : {
                "deter_state": deter_state, #(n*k,Rd)
                "lambdas1" : lambdas1, #self._unflatten_first(lambdas1, k),   #(n*k,Rs)
                "lambdas2" : lambdas2, #self._unflatten_first(lambdas2, k),   #(n*k,Rs)
                "extra_info" : [h1, h2], # Each (1,n*k,lstm_size)
                "samples" : samples      #(n,k,R=Rd+Rs)
            }
        }
        return hidden_state


    #Inputs: hidden_states, images (B,3,D,D), action (B,A) or None, previous_decode_loss_info
    #Outputs: new_hidden_states
    #  Updates posterior lambda but not prior or posterior deter_state
    def refine(self, hidden_states, images, action=None, previous_decode_loss_info=None):
        bs, imsize = images.shape[0], images.shape[2]
        K = self.K
        tiled_k_shape = (bs*K, -1, imsize, imsize)

        if previous_decode_loss_info is None:
            colors, mask, mask_logits = self.decode(hidden_states)  #colors (B,K,3,D,D),  mask (B,K,1,D,D),  mask_logits (B,K,1,D,D)
            color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss = \
                self.get_loss(hidden_states, colors, mask, images)
            #color_probs(B, K, D, D), pixel_complete_log_likelihood(B, D, D), kle_loss(Sc), complete_log_likelihood (Sc), total_loss (Sc)
        else:
            colors, mask, mask_logits = previous_decode_loss_info[0]
            color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss = previous_decode_loss_info[1]

        posterior_mask = color_probs / (color_probs.sum(1, keepdim=True) + 1e-8)  #(B,K,D,D)
        leave_out_ll = pixel_complete_log_likelihood.unsqueeze(1) - mask.squeeze(2) * color_probs #(B,K,D,D)

        x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = \
            torch.autograd.grad(total_loss, [colors, mask, hidden_states["post"]["lambdas1"], hidden_states["post"]["lambdas2"]], create_graph=not self.eval_mode,
                                retain_graph=not self.eval_mode)

        k_images = images.unsqueeze(1).repeat(1, K, 1, 1, 1) #(B,K,3,D,D)

        lns = self.layer_norms
        a = torch.cat([
            torch.cat([k_images.view(tiled_k_shape), colors.view(tiled_k_shape), mask.view(tiled_k_shape), mask_logits.view(tiled_k_shape)], 1), #(B*K,3,D,D), (B*K,3,D,D), (B*K,1,D,D), (B*K,1,D,D) -> (B*K,8,D,D)
            lns[0](torch.cat([
                x_hat_grad.view(tiled_k_shape).detach(), #(B*K,3,D,D)
                mask_grad.view(tiled_k_shape).detach(),  #(B*K,1,D,D)
                posterior_mask.view(tiled_k_shape).detach(), #(B*K,1,D,D)
                pixel_complete_log_likelihood.unsqueeze(1).repeat(1, K, 1, 1).view(tiled_k_shape).detach(), #(B*K,1,D,D)
                leave_out_ll.view(tiled_k_shape).detach()], 1)) #(B*K,1,D,D)
        ], 1) #(B*K,8,D,D), (B*K,7,D,D) -> (B*K,15,D,D)
        # a = torch.cat([
        #     torch.cat([k_images.view(tiled_k_shape), colors.view(tiled_k_shape), mask.view(tiled_k_shape),
        #                mask_logits.view(tiled_k_shape), posterior_mask.view(tiled_k_shape)], 1),
        #     # (B*K,3,D,D), (B*K,3,D,D), (B*K,1,D,D), (B*K,1,D,D), (B*K,1,D,D) -> (B*K,9,D,D)
        #     lns[0](torch.cat([
        #         x_hat_grad.view(tiled_k_shape).detach(),  # (B*K,3,D,D)
        #         mask_grad.view(tiled_k_shape).detach(),  # (B*K,1,D,D)
        #         # posterior_mask.view(tiled_k_shape).detach(),  # (B*K,1,D,D)
        #         pixel_complete_log_likelihood.unsqueeze(1).repeat(1, K, 1, 1).view(tiled_k_shape).detach(),
        #         # (B*K,1,D,D)
        #         leave_out_ll.view(tiled_k_shape).detach()], 1))  # (B*K,1,D,D)
        # ], 1)  # (B*K,9,D,D), (B*K,6,D,D) -> (B*K,15,D,D)

        extra_input = torch.cat([lns[1](lambdas_grad_1.view(bs * K, -1).detach()),
                                 lns[2](lambdas_grad_2.view(bs * K, -1).detach())
                                 ], -1) #(B*K,2*R)

        if action is not None: #Use action as extra input into refinement: This is only for next step refinement (sequence iodine)
            action = self._flatten_first_two(action.unsqueeze(1).repeat(1,K,1)) #(B,A)->(B,K,A)->(B*K,A)

        # h1, h2 = self._flatten_first_two(hidden_states[2]), self._flatten_first_two(hidden_states[3]) #(B*K, R2)
        # h1, h2 = self._flatten_first_two(hidden_states["post"]["extra_info"][0]), \
        #          self._flatten_first_two(hidden_states["post"]["extra_info"][1])  # (B*K, R2)
        h1, h2 = hidden_states["post"]["extra_info"][0], hidden_states["post"]["extra_info"][1] #Each (1,B*K,R2)

        # pdb.set_trace()
        lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2,
                                                         extra_input=torch.cat(
                                                             [extra_input, self._flatten_first_two(hidden_states["post"]["lambdas1"]),
                                                              self._flatten_first_two(hidden_states["post"]["lambdas2"]),
                                                              self._flatten_first_two(hidden_states["post"]["samples"])], -1),
                                                         add_fc_input=action) #Lambdas (B*K,Rs),   h (B*K,R2)

        # new_hidden_states = [self._unflatten_first(lambdas1, K), self._unflatten_first(lambdas2, K), self._unflatten_first(h1, K), self._unflatten_first(h2, K)]

        lambdas1 = self._unflatten_first(lambdas1, K) #(B,K,Rs)
        lambdas2 = self._unflatten_first(lambdas2, K) #(B,K,Rs)
        samples = self.get_samples(lambdas1, lambdas2) #(B,K,Rs)
        new_hidden_states = {
            "prior": hidden_states["prior"], #Do not change prior
            "post": { #Update post
                "deter_state": hidden_states["post"]["deter_state"], #Do not update deterministic part of state (B,K,R)
                "lambdas1": lambdas1, #Update lambdas (B,K,R)
                "lambdas2": lambdas2, #(B,K,R)
                "extra_info": [h1, h2], #Update refinement lstm args, each (1,B*K,R2)
                "samples": samples #Update samples (B,K,R)
            }
        }
        return new_hidden_states

    #Inputs: hidden_states, actions (B,A) or None
    #Outputs: new_hidden_states
    #  Update prior and posterior distribution
    def dynamics(self, hidden_states, actions):
        b, K = hidden_states["post"]["samples"].shape[:2]
        if actions is not None:
            actions = actions.unsqueeze(1).repeat(1, K, 1)  # (B,K,A)
            actions = self._flatten_first_two(actions)  # (B*K,A)
        # samples = self._flatten_first_two(hidden_states["post"]["samples"]) #(B*K,R)
        full_states = self._flatten_first_two(self.get_full_tensor_state(hidden_states)) #(B*K,2R)

        deter_states, lambdas1, lambdas2 = self.dynamics_net(full_states, actions) #Each (B*K,R)
        h1, h2 = ptu.zeros_like(hidden_states["post"]["extra_info"][0]).to(hidden_states["post"]["extra_info"][0].device), \
                 ptu.zeros_like(hidden_states["post"]["extra_info"][1]).to(hidden_states["post"]["extra_info"][1].device) #Set the h's to zero as the next refinement should start from scratch (B,K,R2)

        lambdas1 = self._unflatten_first(lambdas1, K)  # (B,K,R)
        lambdas2 = self._unflatten_first(lambdas2, K)  # (B,K,R)
        samples = self.get_samples(lambdas1, lambdas2)  # (B,K,R)
        new_hidden_states = {
            "prior": { #Update prior
                "lambdas1": lambdas1,  # (B,K,R)
                "lambdas2": lambdas2,  # (B,K,R)
            },
            "post": { #Update posterior
                "deter_state": self._unflatten_first(deter_states, K), #(B,K,R)
                "lambdas1": lambdas1,  # (B,K,R)
                "lambdas2": lambdas2,  # (B,K,R)
                "extra_info": [h1, h2],  # Each (B,K,R2)
                "samples": samples # (B,K,R)
            }
        }
        return new_hidden_states

    #Input: hidden_states
    #Output: colors (B,K,3,D,D),  mask (B,K,1,D,D),  mask_logits (B,K,1,D,D)
    def decode(self, hidden_states):
        bs, k = hidden_states["post"]["samples"].shape[:2]
        full_states = self._flatten_first_two(self.get_full_tensor_state(hidden_states)) #(B*K,2R)
        mask_logits, colors = self.decode_net(full_states) #mask_logits: (B*K,1,D,D),  colors: (B*K,3,D,D)

        mask_logits = self._unflatten_first(mask_logits, k) #(B,K,1,D,D)
        mask = F.softmax(mask_logits, dim=1)  #(B,K,1,D,D), these are the mask probability values
        colors = self._unflatten_first(colors, k) #(B,K,3,D,D)
        # final_recon = (mask * colors).sum(1) #(B,3,D,D)
        # pdb.set_trace()
        return colors, mask, mask_logits


    #Inputs: colors (B,K,3,D,D),  masks (B,K,1,D,D),  target_imgs (B,3,D,D)
    #Outputs: color_probs (B,K,D,D), pixel_complete_log_likelihood (B,D,D), kle_loss (Sc),
    #  complete_log_likelihood (Sc), total_loss (Sc)
    def get_loss(self, hidden_states, colors, mask, target_imgs):
        b, k = colors.shape[:2]
        k_targs = target_imgs.unsqueeze(1).repeat(1, k, 1, 1, 1)  # (B,3,D,D) -> (B,1,3,D,D) -> (B,K,3,D,D)
        k_targs = self._flatten_first_two(k_targs)  # (B,K,3,D,D) -> (B*K,3,D,D)
        tmp_colors = self._flatten_first_two(colors)  # (B,K,3,D,D) -> (B*K,3,D,D)
        color_probs = self._gaussian_prob(tmp_colors, k_targs, self.sigma)  # Computing p(x|h),  (B*K,D,D)
        color_probs = self._unflatten_first(color_probs, k)  # (B,K,D,D)
        pixel_complete_log_likelihood = (mask.squeeze(2) * color_probs).sum(1)  # Sum over K, pixelwise complete log likelihood (B,D,D)
        complete_log_likelihood = -torch.log(pixel_complete_log_likelihood + 1e-12).sum() / b  # (Scalar)

        kle = self.kl_divergence_prior_post(hidden_states["prior"], hidden_states["post"])
        # kle = self.kl_divergence_softplus([hidden_states["post"]["lambdas1"], hidden_states["post"]["lambdas2"]])
        kle_loss = self.beta * kle.sum() / b  # KL loss, (Sc)

        total_loss = complete_log_likelihood + kle_loss  # Total loss, (Sc)
        # pdb.set_trace()
        return color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss


    #Inputs: images: (B, T_obs, 3, D, D),  actions: None or (B, T_acs, A),  initial_hidden_state
    #   schedule: (T1),   loss_schedule:(T1)
    #Output: colors_list (T1,B,K,3,D,D), masks_list (T1,B,K,1,D,D), final_recon (B,3,D,D),
    # total_loss, total_kle_loss, total_clog_prob, mse are all (Sc), end_hidden_state
    def run_schedule(self, images, actions, initial_hidden_state, schedule, loss_schedule):
        b = images.shape[0]
        if initial_hidden_state is None: #Initialize initial_hidden_state if it is not passed in
            initial_hidden_state = self._get_initial_hidden_states(b, images.device)

        #Save outputs: colors_list (T1,B,K,3,D,D),  masks (T1,B,K,1,D,D),  losses_list (T1+1)
        colors_list, masks_list, losses_list, kle_loss_list, clog_prob_list = [], [], [], [], []

        current_step = 0
        cur_hidden_state = initial_hidden_state

        ###Initial loss for initial lambda parameters
        previous_decode_loss_info = None
        # target_images = images[:, current_step]  # (B,3,D,D)
        # colors, mask, mask_logits = self.decode(cur_hidden_state)
        # color_probs, pixel_complete_log_likelihood, kle_loss, clog_prob, total_loss = \
        #     self.get_loss(cur_hidden_state, colors, mask, target_images)
        # losses_list.append(total_loss * loss_schedule[0])
        # kle_loss_list.append(kle_loss * loss_schedule[0])
        # clog_prob_list.append(clog_prob * loss_schedule[0])
        # previous_decode_loss_info = [[colors, mask, mask_logits],
        #                              [color_probs, pixel_complete_log_likelihood, kle_loss, clog_prob, total_loss]]

        ###Loss based on schedule
        for i in range(len(schedule)):
            # print(i)
            if schedule[i] == 0: #Refinement step
                # print("Refinement")
                input_img = images[:, current_step] #(B,3,D,D)
                cur_hidden_state = self.refine(cur_hidden_state, input_img, previous_decode_loss_info=previous_decode_loss_info)
            elif schedule[i] == 1: #Physics step
                # print("Dynamics")
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
                cur_hidden_state = self.refine(cur_hidden_state, input_img, action=input_actions,
                                               previous_decode_loss_info=previous_decode_loss_info)
                current_step += 1
            else:
                raise ValueError("Invalid schedule entry: {}".format(schedule[i]))

            colors, mask, mask_logits = self.decode(cur_hidden_state)
            colors_list.append(colors)
            masks_list.append(mask)

            if loss_schedule[i] != 0:
                target_images = images[:, current_step]  # (B,3,D,D)
                # print("Loss")
                color_probs, pixel_complete_log_likelihood, kle_loss, clog_prob, total_loss = \
                    self.get_loss(cur_hidden_state, colors, mask, target_images)

                losses_list.append(total_loss * loss_schedule[i])
                kle_loss_list.append(kle_loss * loss_schedule[i])
                clog_prob_list.append(clog_prob * loss_schedule[i])

                previous_decode_loss_info = [[colors, mask, mask_logits],
                                      [color_probs, pixel_complete_log_likelihood, kle_loss, clog_prob, total_loss]]
            else:
                previous_decode_loss_info = None

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
        colors_list = colors_list.permute(1, 0, 2, 3, 4, 5) #(T1,B,K,3,D,D) -> (B,T1,K,3,D,D)
        masks_list = masks_list.permute(1, 0, 2, 3, 4, 5) #(T1,B,K,1,D,D) -> (B,T1,K,1,D,D)


        #This part is needed for dataparallel as all tensors need to be (B,*)
        tmp = [cur_hidden_state["post"]["extra_info"][0].squeeze(0), cur_hidden_state["post"]["extra_info"][1].squeeze(0)]
        cur_hidden_state["post"]["extra_info"] = tmp

        # cur_hidden_state = {
        #     "prior": {
        #         "lambdas1": cur_hidden_state["prior"]["lambdas1"].detach(),  # (B,K,R)
        #         "lambdas2": cur_hidden_state["prior"]["lambdas1"].detach(),  # (B,K,R)
        #     },
        #     "post": {  # Update posterior
        #         "deter_state": cur_hidden_state["post"]["deter_state"].detach() if cur_hidden_state["post"]["deter_state"] is not None else None,  # (B,K,R)
        #         "lambdas1": cur_hidden_state["post"]["lambdas1"].detach(),  # (B,K,R)
        #         "lambdas2": cur_hidden_state["post"]["lambdas2"].detach(),  # (B,K,R)
        #         "extra_info": tmp,  # Each (B*K,R2)
        #         "samples": cur_hidden_state["post"]["samples"] # (B,K,R)
        #     }
        # }

        return colors_list, masks_list, final_recon, total_loss, total_kle_loss, total_clog_prob, mse, cur_hidden_state

    #Wrapper for self.run_schedule() required for training with DataParallel
    def forward(self, images, actions, initial_hidden_state, schedule, loss_schedule):
        return self.run_schedule(images, actions, initial_hidden_state, schedule, loss_schedule)


    def get_rprp_schedule(self, seed_steps, num_images, num_refine_per_physics):
        schedule = np.zeros(seed_steps + (num_images-1) * (num_refine_per_physics+1))
        schedule[seed_steps::(num_refine_per_physics+1)] = 1
        return schedule

    def detach_state_info(self, cur_hidden_state):
        tmp = [cur_hidden_state["post"]["extra_info"][0].detach(),
               cur_hidden_state["post"]["extra_info"][1].detach()]

        detached_hidden_state = {
            "prior": {
                "lambdas1": cur_hidden_state["prior"]["lambdas1"].detach(),  # (B,K,R)
                "lambdas2": cur_hidden_state["prior"]["lambdas1"].detach(),  # (B,K,R)
            },
            "post": {  # Update posterior
                "deter_state": cur_hidden_state["post"]["deter_state"].detach() if cur_hidden_state["post"]["deter_state"] is not None else None,
                # (B,K,R)
                "lambdas1": cur_hidden_state["post"]["lambdas1"].detach(),  # (B,K,R)
                "lambdas2": cur_hidden_state["post"]["lambdas2"].detach(),  # (B,K,R)
                "extra_info": tmp,  # Each (B*K,R2)
                "samples": cur_hidden_state["post"]["samples"]  # (B,K,R)
            }
        }
        return detached_hidden_state




