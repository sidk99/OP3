import torch
import torch.utils.data
from rlkit.torch.iodine.physics_network import PhysicsNetwork
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

imsize84_iodine_architecture = dict(
    deconv_args=dict(
        hidden_sizes=[],

        input_width=92,
        input_height=92,
        input_channels=130,

        kernel_sizes=[3, 3, 3, 3],
        n_channels=[64, 64, 64, 64],
        paddings=[0, 0, 0, 0],
        strides=[1, 1, 1, 1],
    ),
    deconv_kwargs=dict(
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    refine_args=dict(
        input_width=84,
        input_height=84,
        input_channels=17,
        paddings=[0, 0, 0, 0],
        kernel_sizes=[3, 3, 3, 3],
        n_channels=[64, 64, 64, 64],
        strides=[2, 2, 2, 2],
        hidden_sizes=[128, 128],
        output_size=128,
        lstm_size=256,
        lstm_input_size=768,
        added_fc_input_size=0

    )
)

imsize64_iodine_architecture = dict(
    deconv_args=dict(
        hidden_sizes=[],

        input_width=80,
        input_height=80,
        input_channels=34,

        kernel_sizes=[5, 5, 5, 5],
        n_channels=[32, 32, 32, 32],
        strides=[1, 1, 1, 1],
        paddings=[0, 0, 0, 0]
    ),
    deconv_kwargs=dict(
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    refine_args=dict(
        input_width=64,
        input_height=64,
        input_channels=17,
        paddings=[0, 0, 0, 0],
        kernel_sizes=[5, 5, 5, 5],
        n_channels=[32, 32, 32, 32],
        strides=[2, 2, 2, 2],
        hidden_sizes=[128, 128],
        output_size=32,
        lstm_size=128,
        lstm_input_size=288,
        added_fc_input_size=0

    )
)

REPSIZE_128 = 128


imsize64_large_iodine_architecture = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=REPSIZE_128,
        input_channels=3,
        # decoder_distribution='gaussian_identity_variance',
        beta=1,
        K=7,
        sigma=0.1
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64*64*3,
        input_width=80,
        input_height=80,
        input_channels=REPSIZE_128 + 2,

        kernel_sizes=[5, 5, 5, 5],
        n_channels=[64, 64, 64, 64],
        strides=[1, 1, 1, 1],
        paddings=[0, 0, 0, 0]
    ),
    deconv_kwargs=dict(
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    refine_args=dict(
        input_width=64,
        input_height=64,
        input_channels=17,
        paddings=[0, 0, 0, 0],
        kernel_sizes=[5, 5, 5, 5],
        n_channels=[64, 64, 64, 64],
        strides=[2, 2, 2, 2],
        hidden_sizes=[128, 128],
        output_size=REPSIZE_128,
        lstm_size=256,
        lstm_input_size=768,
        added_fc_input_size=0

    )
)


def create_model(model, action_dim, dataparallel=False):

    K = model['vae_kwargs']['K']
    rep_size = model['vae_kwargs']['representation_size']

    decoder = BroadcastCNN(**model['deconv_args'], **model['deconv_kwargs'],
                           hidden_activation=nn.ELU())
    refinement_net = RefinementNetwork(**model['refine_args'],
                                       hidden_activation=nn.ELU())
    physics_net = PhysicsNetwork(K, rep_size, action_dim)


    m = IodineVAE(
        **model['vae_kwargs'],
        decoder=decoder,
        refinement_net=refinement_net,
        physics_net=physics_net,
        action_dim=action_dim,
        dataparallel=dataparallel,

    )
    return m

class IodineVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            refinement_net,
            decoder,
            action_dim=None,
            physics_net=None,
            K=3,
            input_channels=1,
            imsize=48,
            min_variance=1e-3,
            beta=5,
            dataparallel=False,
            sigma=0.1,

    ):
        """

        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.K = K
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.refinement_net = refinement_net
        self.decoder_imsize = decoder.input_width
        self.beta = beta
        self.physics_net = physics_net
        self.lstm_size = 256

        self.decoder = decoder

        # if action_dim is not None:
        #     self.action_encoder = Mlp((128,), 32, action_dim,
        #                                  hidden_activation=nn.ELU())
        #     self.action_lambda_encoder = Mlp((256, 256), representation_size, representation_size+32,
        #                                  hidden_activation=nn.ELU())

        l_norm_sizes = [7, 1, 1]
        self.layer_norms = nn.ModuleList([LayerNorm2D(l) for l in l_norm_sizes])

        self.epoch = 0

        self.apply(ptu.init_weights)
        self.lambdas1 = Parameter(ptu.zeros((self.representation_size)))
        self.lambdas2 = Parameter(ptu.ones((self.representation_size)) * 0.6)

        self.sigma = from_numpy(np.array([sigma]))

    def encode(self, input):
        pass

    def decode(self, lambdas1, lambdas2, inputK, bs):

        latents = self.rsample_softplus([lambdas1, lambdas2])

        broadcast_ones = ptu.ones((latents.shape[0], latents.shape[1], self.decoder_imsize, self.decoder_imsize)).to(
            latents.device)
        decoded = self.decoder(latents, broadcast_ones)
        x_hat = decoded[:, :3]
        m_hat_logits = decoded[:, 3]

        m_hat_logit = m_hat_logits.view(bs, self.K, self.imsize, self.imsize)
        mask = F.softmax(m_hat_logit, dim=1)  # (bs, K, imsize, simze)

        pixel_x_prob = self.gaussian_prob(x_hat, inputK, self.sigma).view(bs, self.K, self.imsize, self.imsize)
        pixel_likelihood = (mask * pixel_x_prob).sum(1)  # sum along K
        log_likelihood = -torch.log(pixel_likelihood + 1e-12).sum() / bs

        kle = self.kl_divergence_softplus([lambdas1, lambdas2])
        kle_loss = self.beta * kle.sum() / bs
        loss = log_likelihood + kle_loss

        return x_hat, mask, m_hat_logits, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood

    def gaussian_prob(self, inputs, targets, sigma):
        ch = 3
        # (2pi) ^ ch = 248.05
        sigma = sigma.to(inputs.device)
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (torch.sqrt(sigma**(2 * ch))*248.05)


    def logprob(self, inputs, obs_distribution_params):
        pass

    def gaussian_log_prob(self, inputs, targets, sigma):
        return torch.pow(inputs - targets, 2) / (2 * sigma ** 2)

    def forward(self, input, actions=None, schedule=None, seedsteps=5):

        return self._forward_dynamic_actions(input, actions, schedule)

    def initialize_hidden(self, bs):
        return (ptu.from_numpy(np.zeros((bs, self.lstm_size))),
                ptu.from_numpy(np.zeros((bs, self.lstm_size))))

    def plot_latents(self, ground_truth, masks, x_hats, mse, idx):


        K = self.K
        imsize = self.imsize
        T = len(masks)
        m = torch.stack([m[idx] for m in masks]).permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 3, 1,
                                                                                       1)  # K, T, 3, imsize, imsize
        x = torch.stack(x_hats)[:, K*idx:K*idx+K].permute(1, 0, 2, 3, 4)
        rec = (m * x)
        full_rec = rec.sum(0, keepdim=True)

        comparison = torch.cat([ground_truth, full_rec, m, rec], 0).view(-1, 3, imsize, imsize)

        save_image(comparison.data.cpu(), logger.get_snapshot_dir() + '/goal_latents_%0.5f.png' %mse, nrow=T)

    def plot_latents_trunc(self, ground_truth, masks, x_hats, mse, idx):


        K = self.K
        imsize = self.imsize
        T = len(masks)
        m = torch.stack([m[idx] for m in masks]).permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 3, 1,
                                                                                       1)  # K, T, 3, imsize, imsize
        x = torch.stack(x_hats)[:, K*idx:K*idx+K].permute(1, 0, 2, 3, 4)
        rec = (m * x)
        full_rec = rec.sum(0, keepdim=True)

        comparison = torch.cat([ground_truth, full_rec, m, rec], 0).view(-1, 3, imsize, imsize).data

        #import pdb; pdb.set_trace()
        n_col = 5
        comparison = comparison.view(-1, T, 3, imsize, imsize)[:, -n_col:]
        comparison = comparison[1, ]
        comparison = comparison.contiguous().view(-1, 3, imsize, imsize)

        save_image(comparison.data.cpu(), logger.get_snapshot_dir() + '/goal_latents_%0.5f.png' %mse, nrow=n_col)

    def refine(self, input, hidden_state, plot_latents=False):
        K = self.K
        bs = 8
        input = input.repeat(bs, 1, 1, 1).unsqueeze(1)
        imsize = self.imsize

        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = self._forward_dynamic_actions(input, None,
                                                                                                  schedule=np.zeros((8)))

        recon = x_hats[-1].view(bs, K, 3, imsize, imsize) * masks[-1].unsqueeze(2)
        recon = torch.clamp(recon, 0, 1)
        mse = torch.pow(final_recon - input.squeeze(), 2).mean(3).mean(2).mean(1)
        best_idx = torch.argmin(mse)
        if plot_latents:
            for i in range(8):
                self.plot_latents(input[0].unsqueeze(0).repeat(1, len(masks), 1, 1, 1), masks, x_hats, mse[i], i)
        best_lambda = lambdas[0].view(-1, K, self.representation_size)[best_idx]

        return final_recon[best_idx].data, best_lambda.data, recon[best_idx].data, masks[-1][best_idx].data

    def step(self, input, actions, plot_latents=False):
        K = self.K
        bs = input.shape[0]
        imsize = self.imsize
        input = input.unsqueeze(1)
        schedule = np.ones((12,))
        schedule[:5] = 0

        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = self._forward_dynamic_actions(input, actions,
                                                                                                      schedule=schedule)

        recon = x_hats[-1].view(bs, K, 3, imsize, imsize) * masks[-1].unsqueeze(2)
        recon = torch.clamp(recon, 0, 1)
        if plot_latents:
            i = 0
            self.plot_latents_trunc(input[0].unsqueeze(0).repeat(1, len(masks), 1, 1, 1), masks, x_hats, 0, i)

        return final_recon.data, lambdas[0].view(bs, K, -1).data, recon.data

    def refine_lambdas(self, pixel_x_prob, pixel_likelihood, mask, m_hat_logit, loss, x_hat,
                       lambdas1, lambdas2, inputK, latents, h1, h2, tiled_k_shape, bs):
        K = self.K
        lns = self.layer_norms
        posterior_mask = pixel_x_prob / (pixel_x_prob.sum(1, keepdim=True) + 1e-8)  # avoid divide by zero
        leave_out_ll = pixel_likelihood.unsqueeze(1) - mask * pixel_x_prob
        x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = torch.autograd.grad(loss, [x_hat, mask] + [lambdas1,
                                                                                                           lambdas2],
                                                                                    create_graph=True,
                                                                                    retain_graph=True)

        a = torch.cat([
            torch.cat([inputK, x_hat, mask.view(tiled_k_shape), m_hat_logit.view(tiled_k_shape)], 1),
            lns[0](torch.cat([
                x_hat_grad.detach(),
                mask_grad.view(tiled_k_shape).detach(),
                posterior_mask.view(tiled_k_shape).detach(),
                pixel_likelihood.unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape).detach(),
                leave_out_ll.view(tiled_k_shape).detach()], 1))

        ], 1)

        extra_input = torch.cat([lns[1](lambdas_grad_1.view(bs * K, -1).detach()),
                                 lns[2](lambdas_grad_2.view(bs * K, -1).detach())
                                 ], -1)

        lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2,
                                                         extra_input=torch.cat(
                                                             [extra_input, lambdas1, lambdas2, latents], -1))
        return lambdas1, lambdas2, h1, h2

    def _forward_dynamic_actions(self, input, actions, schedule):
        # input is (bs, T, ch, imsize, imsize)
        # schedule is (T,): 0 for refinement and 1 for physics
        K = self.K
        bs = input.shape[0]
        T = schedule.shape[0]

        # means and log_vars of latent
        lambdas1 = self.lambdas1.unsqueeze(0).repeat(bs*K, 1)
        lambdas2 = self.lambdas2.unsqueeze(0).repeat(bs*K, 1)
        # initialize hidden state
        h1, h2 = self.initialize_hidden(bs*K)

        h1 = h1.to(input.device)
        h2 = h2.to(input.device)

        losses, x_hats, masks = [], [], []
        untiled_k_shape = (bs, K, -1, self.imsize, self.imsize)
        tiled_k_shape = (bs * K, -1, self.imsize, self.imsize)

        current_step = 0

        inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)
        x_hat, mask, m_hat_logit, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood = self.decode(
            lambdas1, lambdas2, inputK, bs)
        losses.append(loss)

        actions_done = False
        applied_action = False

        for t in range(1, T + 1):
            # Refine
            if schedule[t-1] == 0 or actions_done:
                inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)
                lambdas1, lambdas2, h1, h2 = self.refine_lambdas(pixel_x_prob, pixel_likelihood, mask, m_hat_logit, loss, x_hat,
                       lambdas1, lambdas2, inputK, latents, h1, h2, tiled_k_shape, bs)
                # if not applied_action: # Do physics on static scene if haven't applied action yet
                #     lambdas1, _ = self.physics_net(lambdas1, lambdas2, None)
                loss_w = t
            # Physics
            else:
                current_step += 1
                # if current_step == input.shape[1]:
                #     current_step = input.shape[1] - 1
                #     actions_done = True
                applied_action = True
                if actions is not None:
                    actionsK = actions[:, current_step-1].unsqueeze(1).repeat(1, K, 1).view(bs*K, -1)
                else:
                    actionsK = None

                if current_step >= input.shape[1] - 1:
                    actions_done = True
                inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)

                lambdas1, _ = self.physics_net(lambdas1, lambdas2, actionsK)
                loss_w = t * 2



            # Decode and get loss
            x_hat, mask, m_hat_logit, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood = self.decode(
                lambdas1, lambdas2, inputK, bs)

            x_hats.append(x_hat.data)
            masks.append(mask.data)
            losses.append(loss * loss_w)

        total_loss = sum(losses)/T

        final_recon = (mask.unsqueeze(2) * x_hat.view(untiled_k_shape)).sum(1)
        mse = torch.pow(final_recon - input[:, -1], 2).mean()

        all_x_hats = torch.stack([x.view(untiled_k_shape) for x in x_hats], 1) # (bs, T, K, 3, imsize, imsize)
        all_masks = torch.stack([x.view(untiled_k_shape) for x in masks], 1) # # (bs, T, K, 1, imsize, imsize)
        return all_x_hats, all_masks, total_loss, kle_loss.data / self.beta, \
               log_likelihood.data, mse, final_recon.data, [lambdas1.data, lambdas2.data]
