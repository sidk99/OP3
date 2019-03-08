import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from torch.nn.modules.loss import BCEWithLogitsLoss
import numpy as np
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import from_numpy
from rlkit.torch.conv_networks import CNN, DCNN
from rlkit.torch.vae.vae_base import GaussianLatentVAE
from rlkit.torch.modules import LayerNorm2D

imsize84_iodine_architecture = dict(
    deconv_args=dict(
        hidden_sizes=[],

        input_width=92,
        input_height=92,
        input_channels=130,

        kernel_sizes=[3, 3, 3, 3, 1],
        n_channels=[32, 32, 32, 32, 4],
        strides=[1, 1, 1, 1, 1],
    ),
    deconv_kwargs=dict(
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    refine_args=dict(
        input_width=84,
        input_height=84,
        input_channels=19,
        paddings=[0, 0, 0, 0],
        kernel_sizes=[3, 3, 3, 3],
        n_channels=[32, 32, 64, 64],
        strides=[2, 2, 2, 2],
        hidden_sizes=[256, 256],
        output_size=256,
        lstm_size=256,

    )
)

imsize64_iodine_architecture = dict(
    conv_args=dict(
        kernel_sizes=[3, 3, 3, 3],
        n_channels=[32, 32, 64, 64],
        strides=[2, 2, 2, 2],
    ),
    conv_kwargs=dict(
        hidden_sizes=[256, 32],
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    deconv_args=dict(
        hidden_sizes=[],

        input_width=72,
        input_height=72,
        input_channels=18,

        # deconv_output_kernel_size=6,
        # deconv_output_strides=3,
        # deconv_output_channels=3,

        kernel_sizes=[3, 3, 3, 3, 1],
        n_channels=[32, 32, 32, 32, 4],
        strides=[1, 1, 1, 1, 1],
    ),
    deconv_kwargs=dict(
        batch_norm_conv=False,
        batch_norm_fc=False,
    )
)



class IodineVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,
            refinement_net,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
            beta=0.5,
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
        self.save_init_params(locals())
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.refinement_net = refinement_net
        self.beta = beta

        deconv_args, deconv_kwargs = architecture['deconv_args'], architecture['deconv_kwargs']

        self.decoder = decoder_class(
            **deconv_args,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            output_size=self.imlength,
            init_w=init_w,
            hidden_init=hidden_init,
            hidden_activation=nn.ELU(),
            **deconv_kwargs)

        l_norm_sizes = [3, 1, 1, 1, 1]
        self.layer_norms = [LayerNorm2D(l) for l in l_norm_sizes]
        [l.to(ptu.device) for l in self.layer_norms]
        self.epoch = 0
        self.decoder_distribution = decoder_distribution

    def encode(self, input):
        pass

    def decode(self, latents):
        decoded = self.decoder(latents)
        x_hat = decoded[:, :3]
        m_hat_logits = decoded[:, 3]
        return x_hat, torch.ones_like(x_hat), m_hat_logits

    def gaussian_prob(self, inputs, targets, sigma):
        #import pdb; pdb.set_trace()
        #return torch.exp((-(inputs - targets))**2 / (2 * sigma ** 2)) #/ torch.sqrt(2 * sigma**2 * 3.1415)
        return torch.exp(-torch.pow(inputs - targets, 2) / ( 2 * torch.pow(sigma, 2)))

    def gaussian_log_prob(self, inputs, targets, sigma):
        #import pdb; pdb.set_trace()
        #return torch.exp((-(inputs - targets))**2 / (2 * sigma ** 2)) #/ torch.sqrt(2 * sigma**2 * 3.1415)
        return torch.pow(inputs - targets, 2) / (2 * sigma ** 2)

    def cross_entropy_with_logits(self, x, z):
        # x is logits, z is labels
        return torch.max(x, 0)[0] - x * z + torch.log(1 + torch.exp(-torch.abs(x)))

    def logprob(self, inputs, obs_distribution_params):
        pass

    def forward(self, input):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        K = 3
        T = 5
        bs = input.shape[0]
        
        
        input = input.view(bs, 3, self.imsize, self.imsize)
        inputK = torch.cat([input for _ in range(K)], 0) # detach or create copy here?
        sigma = 0.25
        # means and log_vars of latent
        lambdas = [from_numpy(np.zeros((bs, K, self.representation_size))),
                   from_numpy(np.zeros((bs, K, self.representation_size)))]
        # initialize hidden state
        h = self.refinement_net.initialize_hidden(bs*K)
        lns = self.layer_norms
        losses = []
        for t in range(1, T+1):
            z = self.reparameterize(lambdas)
            x_hat, x_var_hat, m_hat_logit = self.decode(z.view(bs*K, self.representation_size))
            m_hat_logit = m_hat_logit.view(bs, K, self.imsize, self.imsize)
            mask = F.softmax(m_hat_logit, dim=1)
            # Retain grads
            mask.retain_grad()
            x_hat.retain_grad()
            [l.retain_grad() for l in lambdas]

            x_prob = self.gaussian_prob(x_hat, inputK, from_numpy(np.array([sigma])))
            likelihood = (mask.unsqueeze(2) * x_prob.view(bs, K, 3, self.imsize, self.imsize))
            log_likelihood = -torch.log(likelihood.sum(1)).sum() / bs
            kle = self.kl_divergence([l.view(bs*K, self.representation_size) for l in lambdas])
            kle_loss = self.beta * kle.sum() / bs
            loss = kle_loss + log_likelihood
            losses.append(t * loss)
            # Compute gradients
            loss.backward(retain_graph=True)

            # Compute inputs a for refinement network
            pixel_likelihood = torch.prod(likelihood, 2) # product among color channels
            posterior_mask = pixel_likelihood / pixel_likelihood.sum(1, keepdim=True)
            leave_out_ll = pixel_likelihood.sum(1, keepdim=True) - pixel_likelihood

            tiled_k_shape = (bs * K, -1, self.imsize, self.imsize)
            a = torch.cat([
                inputK, x_hat, mask.view(tiled_k_shape), m_hat_logit.view(tiled_k_shape),
                lns[0](x_hat.grad.detach()),
                lns[1](mask.grad.view(tiled_k_shape).detach()),
                lns[2](posterior_mask.view(tiled_k_shape)),
                lns[3](pixel_likelihood.unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape).detach()),
                lns[4](leave_out_ll.view(tiled_k_shape).detach())
            ], 1)

            refinement, h = self.refinement_net(a, h)
            refinement = refinement.view(bs, K, self.representation_size*2) # updates for both means and log_vars
            lambdas[0] = lambdas[0] + refinement[:, :, :self.representation_size]
            lambdas[1] = lambdas[1] + refinement[:, :, self.representation_size:]

        total_loss = sum(losses) / T

        return x_hat, mask, total_loss, kle_loss, log_likelihood



