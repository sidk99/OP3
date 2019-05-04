import torch
import torch.utils.data
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
from rlkit.torch.conv_networks import CNN, DCNN
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

imsize64_large_iodine_architecture = dict(
    deconv_args=dict(
        hidden_sizes=[],

        input_width=80,
        input_height=80,
        input_channels=130,

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
        output_size=128,
        lstm_size=256,
        lstm_input_size=768,
        added_fc_input_size=0

    )
)


class IodineVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,
            refinement_net,
            physics_net=None,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            K=3,
            T=5,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
            beta=5,
            dynamic=False,
            dataparallel=False,
            sigma=0.1,
    ):
        """

        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
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
        self.T = T
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.refinement_net = refinement_net
        self.beta = beta
        self.dynamic = dynamic
        self.physics_net = physics_net
        self.lstm_size = 256
        deconv_args, deconv_kwargs = architecture['deconv_args'], architecture['deconv_kwargs']

        self.decoder_imsize = deconv_args['input_width']
        self.decoder = decoder_class(
            **deconv_args,
            output_size=self.imlength,
            init_w=init_w,
            hidden_init=hidden_init,
            hidden_activation=nn.ELU(),
            **deconv_kwargs)

        self.action_encoder = Mlp((128,), 128, 13,
                                     hidden_activation=nn.ELU())
        self.action_lambda_encoder = Mlp((256, 256), representation_size, representation_size+128,
                                     hidden_activation=nn.ELU())

        l_norm_sizes = [7, 1, 1]
        self.layer_norms = nn.ModuleList([LayerNorm2D(l) for l in l_norm_sizes])

        if dataparallel:
            self.decoder = nn.DataParallel(self.decoder)
            #self.physics_net = nn.DataParallel(self.physics_net)
            self.refinement_net = nn.DataParallel(self.refinement_net)


        self.epoch = 0
        self.decoder_distribution = decoder_distribution

        self.apply(ptu.init_weights)
        self.lambdas = nn.ParameterList([Parameter(ptu.zeros((1, self.representation_size))),
                        Parameter(ptu.ones((1, self.representation_size)) * 0.6)]) #+ torch.exp(ptu.ones((1, self.representation_size)))))]

        self.sigma = from_numpy(np.array([sigma]))

    def encode(self, input):
        pass

    def decode(self, latents):
        broadcast_ones = ptu.ones((latents.shape[0], latents.shape[1], self.decoder_imsize, self.decoder_imsize))
        decoded = self.decoder(latents, broadcast_ones)
        x_hat = decoded[:, :3]
        m_hat_logits = decoded[:, 3]
        return x_hat, torch.ones_like(x_hat), m_hat_logits

    def gaussian_prob(self, inputs, targets, sigma):
        ch = 3
        # (2pi) ^ ch = 248.05
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (torch.sqrt(sigma**(2 * ch))*248.05)


    def gaussian_log_prob(self, inputs, targets, sigma):
        return torch.pow(inputs - targets, 2) / (2 * sigma ** 2)

    def cross_entropy_with_logits(self, x, z):
        # x is logits, z is labels
        return torch.max(x, 0)[0] - x * z + torch.log(1 + torch.exp(-torch.abs(x)))

    def logprob(self, inputs, obs_distribution_params):
        pass

    def forward(self, input, actions=None, schedule=None, seedsteps=5):
        if actions is None:
            return self._forward_dynamic(input, seedsteps=seedsteps)
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

    def _forward_dynamic_actions(self, input, actions, schedule):
        # input is (bs, T, ch, imsize, imsize)
        # schedule is (T,): 0 for refinement and 1 for physics
        K = self.K
        bs = input.shape[0]
        T = schedule.shape[0]

        # means and log_vars of latent
        lambdas = [l.repeat(bs * K, 1) for l in self.lambdas]

        # initialize hidden state
        h1, h2 = self.initialize_hidden(bs*K)
        lns = self.layer_norms
        losses = []
        x_hats = []
        masks = []

        # choose random latent to contain action by zeroing out others
        if actions is not None:
            actionsK = actions.unsqueeze(1).repeat(1, K, 1)
            action_mask = ptu.zeros(bs, K)
            action_mask[np.arange(0, bs), np.random.randint(0, K, bs)] = 1
            #actionsK *= action_mask.unsqueeze(-1)
            actionsK = actionsK.view(bs*K, -1)


        untiled_k_shape = (bs, K, -1, self.imsize, self.imsize)
        tiled_k_shape = (bs * K, -1, self.imsize, self.imsize)


        current_step = 0
        apply_action = True

        for t in range(1, T + 1):
            if schedule[t-1] == 0:
                z = self.rsample_softplus(lambdas)
                x_hat, x_var_hat, m_hat_logit = self.decode(z)  # x_hat is (bs*K, 3, imsize, imsize)
                # x_hat = torch.clamp(x_hat, 0, 1) # doing this clamping causes the alternating mask issue
                m_hat_logit = m_hat_logit.view(bs, K, self.imsize, self.imsize)
                mask = F.softmax(m_hat_logit, dim=1)  # (bs, K, imsize, simze)
                x_hats.append(x_hat)
                masks.append(mask)
                inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)
                pixel_x_prob = self.gaussian_prob(x_hat, inputK, self.sigma).view(bs, K, self.imsize, self.imsize)
                pixel_likelihood = (mask * pixel_x_prob).sum(1)  # sum along K
                log_likelihood = -torch.log(pixel_likelihood + 1e-12).sum() / bs

                kle = self.kl_divergence_softplus(lambdas)
                kle_loss = self.beta * kle.sum() / bs
                loss = log_likelihood + kle_loss
                losses.append(loss/t)

                # Compute inputs a for refinement network
                posterior_mask = pixel_x_prob / (pixel_x_prob.sum(1, keepdim=True) + 1e-8)  # avoid divide by zero
                leave_out_ll = pixel_likelihood.unsqueeze(1) - mask * pixel_x_prob
                x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = torch.autograd.grad(loss, [x_hat, mask] + lambdas,
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
                extra_input = torch.cat([extra_input, lambdas[0], lambdas[1], z], -1)

                lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2, extra_input=extra_input)

                # If haven't applied action yet and still seeding then also do physics on seed images
                if apply_action:
                   lambdas1, _ = self.physics_net(lambdas1, lambdas2)

            else:
                z = self.rsample_softplus(lambdas)
                x_hat, x_var_hat, m_hat_logit = self.decode(z)
                # x_hat = torch.clamp(x_hat, 0, 1) # doing this clamping causes the alternating mask issue
                m_hat_logit = m_hat_logit.view(bs, K, self.imsize, self.imsize)
                mask = F.softmax(m_hat_logit, dim=1)  # (bs, K, imsize, simze)
                x_hats.append(x_hat)
                masks.append(mask)

                inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)
                pixel_x_prob = self.gaussian_prob(x_hat, inputK, self.sigma).view(bs, K, self.imsize, self.imsize)
                pixel_likelihood = (mask * pixel_x_prob).sum(1)  # sum along K
                log_likelihood = -torch.log(pixel_likelihood + 1e-12).sum() / bs

                kle = self.kl_divergence_softplus(lambdas)
                kle_loss = self.beta * kle.sum() / bs
                loss = log_likelihood + kle_loss
                losses.append(loss/t)

                current_step += 1
                current_step = min(input.shape[1]-1, current_step)
                if apply_action:
                    action_enc = self.action_encoder(actionsK)
                    lambdas1 = self.action_lambda_encoder(torch.cat([lambdas1, action_enc], -1))
                    apply_action = False
                lambdas1, _ = self.physics_net(lambdas1, lambdas2)


            lambdas[0] = lambdas1
            lambdas[1] = lambdas2

        total_loss = sum(losses*T)

        final_recon = (mask.unsqueeze(2) * x_hat.view(untiled_k_shape)).sum(1)
        mse = torch.pow(final_recon - input[:, current_step], 2).mean()

        return x_hats, masks, total_loss, kle_loss / self.beta, log_likelihood, mse, final_recon, lambdas
