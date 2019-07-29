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
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (np.sqrt(sigma ** (2 * ch)) * 248.05)
        # sigma = sigma.to(inputs.device)
        # return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (torch.sqrt(sigma ** (2 * ch)) * 248.05)

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


    #Compute the kl loss between prior and posterior
    def kl_divergence_prior_post(self, prior, post):
        mu1, softplus1 = prior["lambdas1"], prior["lambdas2"]
        mu2, softplus2 = post["lambdas1"], post["lambdas2"]

        stds1 = torch.sqrt(torch.log(1 + softplus1.exp()))
        stds2 = torch.sqrt(torch.log(1 + softplus2.exp()))
        # print(mu1.device, softplus1.device, stds1.device, mu2.device, softplus2.device, stds2.device)
        # print(mu1.shape, softplus1.shape, stds1.shape, mu2.shape, softplus2.shape, stds2.shape)
        q1 = MultivariateNormal(loc=mu1, scale_tril=torch.diag_embed(stds1.to(mu1.device)))
        q2 = MultivariateNormal(loc=mu2, scale_tril=torch.diag_embed(stds2.to(mu2.device)))

        # q1 = Normal(loc=mu1, scale=stds1)
        # q2 = Normal(loc=mu2, scale=stds2)

        return torch.distributions.kl.kl_divergence(q2, q1) #KL(post||prior), note ordering matters!
        # return mu1 + mu2 + stds1 + stds2

    def get_samples(self, means, softplusses):
        stds = torch.sqrt(torch.log(1 + softplusses.exp()))
        epsilon = ptu.randn(*means.size()).to(stds.device)  # RV: Is the star necessary?
        latents = epsilon * stds + means
        return latents

    #Input: Integer n denoting how many hidden states to initialize
    #Output: Returns hidden states, tuples of (n,k,repsize) and (n,k,lstm_size)
    def _get_initial_hidden_states(self, n, device):
        k = self.K
        # lambdas1 = self._unflatten_first(self.lambdas1.unsqueeze(0).repeat(n*k, 1), k) #(n,k,repsize)
        # lambdas2 = self._unflatten_first(self.lambdas2.unsqueeze(0).repeat(n*k, 1), k) #(n,k,repsize)
        # h1, h2 = self.refinement_net.initialize_hidden(n*k) #Each (n*k,lstm_size)
        # h1, h2 = self._unflatten_first(h1, k), self._unflatten_first(h2, k) #Each (n,k,lstm_size)
        # return [lambdas1, lambdas2, h1, h2]

        lambdas1 = self.lambdas1.unsqueeze(0).repeat(n*k, 1) #(n*k,repsize)
        lambdas2 = self.lambdas2.unsqueeze(0).repeat(n*k, 1) #(n*k,repsize)
        samples = self._unflatten_first(self.get_samples(lambdas1, lambdas2), k) #(n,k,repsize)
        h1, h2 = self.refinement_net.initialize_hidden(n * k)  # Each (n*k,lstm_size)
        h1, h2 = h1.to(device), h2.to(device)
        # h1, h2 = self._unflatten_first(h1, k), self._unflatten_first(h2, k)  # Each (n,k,lstm_size)

        hidden_state = {
            "prior" : {
                "lambdas1": ptu.zeros_like(self._unflatten_first(lambdas1, k)).to(device),  # (n,k,repsize)
                "lambdas2": ptu.ones_like(self._unflatten_first(lambdas2, k)).to(device),  # (n,k,repsize)
            },
            "post" : {
                "lambdas1" : self._unflatten_first(lambdas1, k),   #(n,k,repsize)
                "lambdas2" : self._unflatten_first(lambdas2, k),   #(n,k,repsize)
                "extra_info" : [h1, h2], # Each (n*k,lstm_size)
                "samples" : samples      #(n,k,repsize)
            }
        }
        return hidden_state


    #Inputs: images: (B, 3, D, D),  hidden_states: Tuples of (B, K, R),  action: None or (B, A)
    #Outputs: new_hidden_states: Tuples of (B, K, R)
    #  Updates posterior but not prior
    def refine(self, hidden_states, images, action=None):
        bs, imsize = images.shape[0], images.shape[2]
        K = self.K
        tiled_k_shape = (bs*K, -1, imsize, imsize)

        # lambdas1, lambdas2 = self._flatten_first_two(hidden_states[0]), self._flatten_first_two(hidden_states[1])  # Each (B*K, R)
        # lambdas1, lambdas2 = self._flatten_first_two(hidden_states["post"]["lambdas1"]), \
        #                      self._flatten_first_two(hidden_states["post"]["lambdas2"]) # Each (B*K, R)

        # colors, mask, mask_logits, sampled_latents, color_probs, pixel_complete_log_likelihood, kle_loss, total_loss, complete_log_likelihood\
        #     = self.decode(hidden_states, images)
        colors, mask, mask_logits = self.decode(hidden_states)  #colors (B,K,3,D,D),  mask (B,K,1,D,D),  mask_logits (B,K,1,D,D)
        color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss = \
            self.get_loss(hidden_states, colors, mask, images)

        posterior_mask = color_probs / (color_probs.sum(1, keepdim=True) + 1e-8)  #(B,K,D,D)
        leave_out_ll = pixel_complete_log_likelihood.unsqueeze(1) - mask.squeeze(2) * color_probs #(B,K,D,D)

        # pdb.set_trace()
        x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = \
            torch.autograd.grad(total_loss, [colors, mask, hidden_states["post"]["lambdas1"], hidden_states["post"]["lambdas2"]], create_graph=not self.eval_mode,
                                retain_graph=not self.eval_mode)
        # lambdas_grad_1 = self._flatten_first_two(lambdas_grad_1) #(B*K,R)
        # lambdas_grad_2 = self._flatten_first_two(lambdas_grad_2)  # (B*K,R)

        k_images = images.unsqueeze(1).repeat(1, K, 1, 1, 1) #(B,K,3,D,D)

        lns = self.layer_norms
        a = torch.cat([
            torch.cat([k_images.view(tiled_k_shape), colors.view(tiled_k_shape), mask.view(tiled_k_shape), mask_logits.view(tiled_k_shape)], 1),
            lns[0](torch.cat([
                x_hat_grad.view(tiled_k_shape).detach(),
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

        # h1, h2 = self._flatten_first_two(hidden_states[2]), self._flatten_first_two(hidden_states[3]) #(B*K, R2)
        # h1, h2 = self._flatten_first_two(hidden_states["post"]["extra_info"][0]), \
        #          self._flatten_first_two(hidden_states["post"]["extra_info"][1])  # (B*K, R2)
        h1, h2 = hidden_states["post"]["extra_info"][0], hidden_states["post"]["extra_info"][1] #(B*K, R2)
        lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2,
                                                         extra_input=torch.cat(
                                                             [extra_input, self._flatten_first_two(hidden_states["post"]["lambdas1"]),
                                                              self._flatten_first_two(hidden_states["post"]["lambdas2"]),
                                                              self._flatten_first_two(hidden_states["post"]["samples"])], -1),
                                                         add_fc_input=action) #Lambdas (B*K,R),   h (B*K,R2)

        # new_hidden_states = [self._unflatten_first(lambdas1, K), self._unflatten_first(lambdas2, K), self._unflatten_first(h1, K), self._unflatten_first(h2, K)]
        new_hidden_states = {
            "prior": hidden_states["prior"], #Do not change prior
            "post": { #Update post
                "lambdas1": self._unflatten_first(lambdas1, K), #(B,K,R)
                "lambdas2": self._unflatten_first(lambdas2, K), #(B,K,R)
                "extra_info": [h1, h2], #[self._unflatten_first(h1, K), self._unflatten_first(h2, K)], #Each (B,K,R2)
                "samples": self._unflatten_first(self.get_samples(lambdas1, lambdas2), K) #(B,K,R)
            }
        }
        return new_hidden_states

    #Inputs: hidden_states,  actions (B, A)
    #Outputs: new_hidden_states
    #  Update prior and posterior distribution
    def dynamics(self, hidden_states, actions):
        # bs, k = hidden_states[0].shape[:2] #hidden_states[0] is lambdas1 (B,K,R)
        # lambdas1, lambdas2 = self._flatten_first_two(hidden_states[0]), self._flatten_first_two(hidden_states[1]) #Each (B*K,R)
        # actions = actions.unsqueeze(1).repeat(1, k, 1) #(B,K,A)
        # actions = self._flatten_first_two(actions) #(B*K,A)
        # lambdas1, lambdas2 = self.dynamics_net(lambdas1, lambdas2, actions) #(B*K,R)
        # lambdas1 = self._unflatten_first(lambdas1, k) #(B,K,R)
        # lambdas2 = self._unflatten_first(lambdas2, k) #(B,K,R)
        #
        # h1, h2 = ptu.zeros_like(hidden_states[2]), ptu.zeros_like(hidden_states[3]) #Set the h's to zero as the next refinement should start from scratch (B,K,R2)
        # return [lambdas1, lambdas2, h1, h2]
        b, K = hidden_states["post"]["samples"].shape[:2]
        actions = actions.unsqueeze(1).repeat(1, K, 1)  # (B,K,A)
        actions = self._flatten_first_two(actions)  # (B*K,A)
        samples = self._flatten_first_two(hidden_states["post"]["samples"]) #(B*K,R)
        lambdas1, lambdas2 = self.dynamics_net(samples, actions) #Each (B*K,R)

        h1, h2 = ptu.zeros_like(hidden_states["post"]["extra_info"][0]).to(hidden_states["post"]["extra_info"][0].device), \
                 ptu.zeros_like(hidden_states["post"]["extra_info"][1]).to(hidden_states["post"]["extra_info"][1].device) #Set the h's to zero as the next refinement should start from scratch (B,K,R2)

        new_hidden_states = {
            "prior": { #New prior is old posterior
                "lambdas1": hidden_states["post"]["lambdas1"],
                "lambdas2": hidden_states["post"]["lambdas2"]
            },
            "post": { #Update posterior
                "lambdas1": self._unflatten_first(lambdas1, K),  # (B,K,R)
                "lambdas2": self._unflatten_first(lambdas2, K),  # (B,K,R)
                "extra_info": [h1, h2],  # Each (B,K,R2)
                "samples": self._unflatten_first(self.get_samples(lambdas1, lambdas2), K)  # (B,K,R)
            }
        }
        return new_hidden_states

    #Inputs: hidden_states: Tuples of (B, K, R), target_imgs: None or (B, 3, D, D)
    #Outputs:
    #  If target_imgs is None: colors (B,K,3,D,D), masks (B,K,1,D,D)
    #  Else, compute the relevant losses as well:
    #  colors (B,K,3,D,D), masks (B,K,1,D,D), mask_logits (B,K,1,D,D), sampled_latents (B,K,R), color_probs (B,K,D,D)
    #  pixel_complete_log_likelihood (B,D,D), kle_loss (Sc), loss (Sc), complete_log_likelihood (Sc)
    #    Note: The outputted losses are normalized by batch size
    # def decode(self, hidden_states, target_imgs=None):
    #     # bs, k = hidden_states[0].shape[:2] #hidden_states[0] is lambdas1 (B,K,R)
    #     # lambdas1, lambdas2 = self._flatten_first_two(hidden_states[0]), self._flatten_first_two(hidden_states[1]) #Each (B*K, R)
    #
    #     # sampled_latents = self.rsample_softplus([lambdas1, lambdas2]) #lambdas1, lambdas2 are mu, softplus;  sampled_latents: (B*K,R)
    #     bs, k = hidden_states["samples"].shape[:2]
    #     sampled_latents = self._flatten_first_two(hidden_states["samples"]) #(B,K,R)
    #     mask_logits, colors = self.decode_net(sampled_latents) #mask_logits: (B*K,1,D,D),  colors: (B*K,3,D,D)
    #
    #     mask_logits = self._unflatten_first(mask_logits, k) #(B,K,1,D,D)
    #     mask = F.softmax(mask_logits, dim=1)  #(B,K,1,D,D), these are the mask probability values
    #     colors = self._unflatten_first(colors, k) #(B,K,3,D,D)
    #     # final_recon = (mask * colors).sum(1) #(B,3,D,D)
    #
    #     if target_imgs is not None:
    #         k_targs = target_imgs.unsqueeze(1).repeat(1, k, 1, 1, 1) #(B,3,D,D) -> (B,1,3,D,D) -> (B,K,3,D,D)
    #         k_targs = self._flatten_first_two(k_targs) #(B,K,3,D,D) -> (B*K,3,D,D)
    #         tmp_colors = self._flatten_first_two(colors) #(B,K,3,D,D) -> (B*K,3,D,D)
    #         color_probs = self._gaussian_prob(tmp_colors, k_targs, self.sigma) #Computing p(x|h),  (B*K,D,D)
    #         color_probs = self._unflatten_first(color_probs, k) #(B,K,D,D)
    #         pixel_complete_log_likelihood = (mask.squeeze(2)*color_probs).sum(1) #Sum over K, pixelwise complete log likelihood (B,D,D)
    #         complete_log_likelihood = -torch.log(pixel_complete_log_likelihood + 1e-12).sum()/bs #(Scalar)
    #
    #         kle = self.kl_divergence_softplus([lambdas1, lambdas2])
    #         kle_loss = self.beta * kle.sum() / bs #KL loss, (Sc)
    #
    #         total_loss = complete_log_likelihood + kle_loss #Total loss, (Sc)
    #         sampled_latents = self._unflatten_first(sampled_latents, k) #(B,K,R)
    #         return colors, mask, mask_logits, sampled_latents, color_probs, pixel_complete_log_likelihood, kle_loss, total_loss, complete_log_likelihood
    #     else:
    #         return colors, mask

    #Input: hidden_states
    #Output: colors (B,K,3,D,D),  mask (B,K,1,D,D),  mask_logits (B,K,1,D,D)
    def decode(self, hidden_states):
        bs, k = hidden_states["post"]["samples"].shape[:2]
        sampled_latents = self._flatten_first_two(hidden_states["post"]["samples"]) #(B*K,R)
        mask_logits, colors = self.decode_net(sampled_latents) #mask_logits: (B*K,1,D,D),  colors: (B*K,3,D,D)

        mask_logits = self._unflatten_first(mask_logits, k) #(B,K,1,D,D)
        mask = F.softmax(mask_logits, dim=1)  #(B,K,1,D,D), these are the mask probability values
        colors = self._unflatten_first(colors, k) #(B,K,3,D,D)
        # final_recon = (mask * colors).sum(1) #(B,3,D,D)
        return colors, mask, mask_logits

        # if target_imgs is not None:
        #     k_targs = target_imgs.unsqueeze(1).repeat(1, k, 1, 1, 1) #(B,3,D,D) -> (B,1,3,D,D) -> (B,K,3,D,D)
        #     k_targs = self._flatten_first_two(k_targs) #(B,K,3,D,D) -> (B*K,3,D,D)
        #     tmp_colors = self._flatten_first_two(colors) #(B,K,3,D,D) -> (B*K,3,D,D)
        #     color_probs = self._gaussian_prob(tmp_colors, k_targs, self.sigma) #Computing p(x|h),  (B*K,D,D)
        #     color_probs = self._unflatten_first(color_probs, k) #(B,K,D,D)
        #     pixel_complete_log_likelihood = (mask.squeeze(2)*color_probs).sum(1) #Sum over K, pixelwise complete log likelihood (B,D,D)
        #     complete_log_likelihood = -torch.log(pixel_complete_log_likelihood + 1e-12).sum()/bs #(Scalar)
        #
        #     kle = self.kl_divergence_softplus([lambdas1, lambdas2])
        #     kle_loss = self.beta * kle.sum() / bs #KL loss, (Sc)
        #
        #     total_loss = complete_log_likelihood + kle_loss #Total loss, (Sc)
        #     sampled_latents = self._unflatten_first(sampled_latents, k) #(B,K,R)
        #     return colors, mask, mask_logits, sampled_latents, color_probs, pixel_complete_log_likelihood, kle_loss, total_loss, complete_log_likelihood
        # else:
        #     return colors, mask

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
        return color_probs, pixel_complete_log_likelihood, kle_loss, complete_log_likelihood, total_loss


    #Inputs: images: (B, T_obs, 3, D, D),  actions: (B, T_acs, A),  initial_hidden_state: Tuples of (B, K, repsize)
    #   schedule: (T1),   loss_schedule:(T1)
    #Output: colors_list (T1,B,K,3,D,D), masks_list (T1,B,K,1,D,D), final_recon (B,3,D,D),
    # total_loss, total_kle_loss, total_clog_prob, mse are all (Sc)
    def run_schedule(self, images, actions, initial_hidden_state, schedule, loss_schedule):
        # pdb.set_trace()
        b = images.shape[0]
        if initial_hidden_state is None: #Initialize initial_hidden_state if it is not passed in
            initial_hidden_state = self._get_initial_hidden_states(b, images.device)

        #Save outputs: colors_list (T1,B,K,3,D,D),  masks (T1,B,K,1,D,D),  losses_list (T1)
        colors_list, masks_list, losses_list, kle_loss_list, clog_prob_list = [], [], [], [], []

        current_step = 0
        cur_hidden_state = initial_hidden_state
        # print("Starting for loop!")
        for i in range(len(schedule)):
            # print(i)
            if schedule[i] == 0: #Refinement step
                # print("Refinement")
                input_img = images[:, current_step] #(B,3,D,D)
                cur_hidden_state = self.refine(cur_hidden_state, input_img)
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
                cur_hidden_state = self.refine(cur_hidden_state, input_img, action=input_actions)
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
        # print("Out of for loop!")

        colors_list = torch.stack(colors_list) #(T1,B,K,3,D,D)
        masks_list = torch.stack(masks_list) #(T1,B,K,1,D,D)

        if sum(loss_schedule) == 0:
            total_loss, total_kle_loss, total_clog_prob = None, None, None
            # print("No loss!")
        else:
            sum_loss_weights = sum(loss_schedule)
            total_loss = sum(losses_list) / sum_loss_weights #Scalar
            total_kle_loss = sum(kle_loss_list) / sum_loss_weights
            total_clog_prob = sum(clog_prob_list) / sum_loss_weights

        final_recon = (colors_list[-1] * masks_list[-1]).sum(1) #(B,K,3,D,D) -> (B,3,D,D)
        mse = torch.pow(final_recon - images[:, -1], 2).mean() #(B,3,D,D) -> (Sc)
        colors_list = colors_list.permute(1, 0, 2, 3, 4, 5) #(T1,B,K,3,D,D) -> (B,T1,K,3,D,D)
        masks_list = masks_list.permute(1, 0, 2, 3, 4, 5) #(T1,B,K,1,D,D) -> (B,T1,K,1,D,D)

        return colors_list, masks_list, final_recon, total_loss, total_kle_loss, total_clog_prob, mse

    #Wrapper for self.run_schedule() required for training with DataParallel
    def forward(self, images, actions, initial_hidden_state, schedule, loss_schedule):
        return self.run_schedule(images, actions, initial_hidden_state, schedule, loss_schedule)
