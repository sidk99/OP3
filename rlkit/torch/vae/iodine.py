import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F, Parameter
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

        kernel_sizes=[3, 3, 3, 3],
        n_channels=[64, 64, 64, 64],
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



class IodineVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,
            refinement_net,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            K=3,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
            beta=5,
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
        self.K = K
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.refinement_net = refinement_net
        self.beta = beta

        deconv_args, deconv_kwargs = architecture['deconv_args'], architecture['deconv_kwargs']

        self.decoder = decoder_class(
            **deconv_args,
            output_size=self.imlength,
            init_w=init_w,
            hidden_init=hidden_init,
            hidden_activation=nn.ELU(),
            **deconv_kwargs)

        l_norm_sizes = [7, 1, 1]
        self.layer_norms = nn.ModuleList([LayerNorm2D(l) for l in l_norm_sizes])
        [l.to(ptu.device) for l in self.layer_norms]
        self.epoch = 0
        self.decoder_distribution = decoder_distribution

        self.apply(ptu.init_weights)
        self.lambdas = [Parameter(ptu.zeros((1, self.representation_size))),
                        Parameter(ptu.ones((1, self.representation_size)) * 0.6)] #+ torch.exp(ptu.ones((1, self.representation_size)))))]

    def encode(self, input):
        pass

    def decode(self, latents):
        decoded = self.decoder(latents)
        x_hat = decoded[:, :3]
        m_hat_logits = decoded[:, 3]
        return x_hat, torch.ones_like(x_hat), m_hat_logits

    def gaussian_prob(self, inputs, targets, sigma):
        #import pdb; pdb.set_trace()
        ch = 3
        # (2pi) ^ ch = 248.05
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (torch.sqrt(sigma**(2 * ch))*248.05)
        #var = sigma ** 2
        #return torch.prod(1/torch.sqrt(2 * 3.1415*var) * torch.exp(-torch.pow(inputs-targets, 2) / ( 2 * var)), 1)
        #return torch.exp(-torch.pow(inputs - targets, 2) / ( 2 * torch.pow(sigma, 2)))

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
        K = self.K
        T = 5
        bs = input.shape[0]
        
        
        input = input.view(bs, 3, self.imsize, self.imsize)
        inputK = input.unsqueeze(1).repeat(1, K, 1, 1, 1).view(bs*K, 3, self.imsize, self.imsize)


        sigma = 0.1
        # means and log_vars of latent
        lambdas = [l.repeat(bs*K, 1) for l in self.lambdas]

        # initialize hidden state
        h = self.refinement_net.initialize_hidden(bs*K)
        lns = self.layer_norms
        losses = []
        x_hats = []
        masks = []
        for t in range(1, T+1):
            z = self.rsample_softplus(lambdas)
            x_hat, x_var_hat, m_hat_logit = self.decode(z) # x_hat is (bs*K, 3, imsize, imsize)
            #import pdb; pdb.set_trace()
            m_hat_logit = m_hat_logit.view(bs, K, self.imsize, self.imsize)
            mask = F.softmax(m_hat_logit, dim=1)
            x_hats.append(x_hat)
            masks.append(mask)


            pixel_x_prob = self.gaussian_prob(x_hat, inputK, from_numpy(np.array([sigma]))).view(bs, K, self.imsize, self.imsize)
            pixel_likelihood = (mask * pixel_x_prob).sum(1) # sum along K
            log_likelihood = -torch.log(pixel_likelihood + 1e-30).sum() / bs
            # pixel_x_prob = torch.pow(x_hat - inputK, 2).mean(1).view(bs, K, self.imsize, self.imsize)
            # pixel_likelihood = (mask * pixel_x_prob).sum(1)
            # log_likelihood = pixel_likelihood.sum() / bs / sigma ** 2

            kle = self.kl_divergence_softplus(lambdas)
            kle_loss = self.beta * kle.sum() / bs
            loss = log_likelihood + kle_loss
            losses.append(t * loss)
            #losses.append(loss)

            # Compute inputs a for refinement network
            posterior_mask = pixel_x_prob / (pixel_x_prob.sum(1, keepdim=True) + 1e-8) # avoid divide by zero
            leave_out_ll = pixel_likelihood.unsqueeze(1) - mask * pixel_x_prob
            x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = torch.autograd.grad(loss, [x_hat, mask] + lambdas,
                                                                                        create_graph=True, retain_graph=True)

            tiled_k_shape = (bs * K, -1, self.imsize, self.imsize) # TODO when collapsing back to this shape is K still most contiguous?

            # TODO clip all grads during T
            a = torch.cat([
                    torch.cat([inputK, x_hat, mask.view(tiled_k_shape), m_hat_logit.view(tiled_k_shape)], 1),
                    lns[0](torch.cat([
                        x_hat_grad.detach(),
                        mask_grad.view(tiled_k_shape).detach(),
                        posterior_mask.view(tiled_k_shape).detach(),
                        pixel_likelihood.unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape).detach(),
                        leave_out_ll.view(tiled_k_shape).detach()], 1))

                ], 1)

            extra_input = torch.cat([lns[1](lambdas_grad_1.view(bs*K, -1).detach()),
                                    lns[2](lambdas_grad_2.view(bs*K, -1).detach())
                                    ], -1)
            #extra_input = lns[1](lambdas_grad_1.view(bs * K, -1).detach())
            extra_input = torch.cat([extra_input, lambdas[0], lambdas[1], z], -1)
            refinement1, refinement2, h = self.refinement_net(a, h, extra_input=extra_input)
            #import pdb; pdb.set_trace()
            lambdas[0] = refinement1
            lambdas[1] = refinement2

        total_loss = sum(losses) / T
        #total_loss = losses[-1]

        final_recon = (mask.unsqueeze(2) * x_hat.view(bs, K, 3, self.imsize, self.imsize)).sum(1)
        mse = torch.pow(final_recon - input, 2).mean()


        return x_hats, masks, total_loss, kle_loss / self.beta, log_likelihood, mse



