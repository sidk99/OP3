from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from torch.nn.modules.loss import BCEWithLogitsLoss
import numpy as np
from rlkit.torch.pytorch_util import from_numpy
from rlkit.torch.conv_networks import CNN, DCNN
from rlkit.torch.vae.vae_base import GaussianLatentVAE
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

np.random.seed(0)
torch.manual_seed(1)

list_architecture = dict(
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

model_specs = dict(
    layer1_args= dict(
        fc1 = [4,64],
        relu= True
    )
)






#parser = argparse.ArgumentParser(description='VAE MNIST Example')
#parser.add_argument('--batch-size', type=int, default=128, metavar='N',
  #                  help='input batch size for training (default: 128)')
#parser.add_argument('--epochs', type=int, default=10, metavar='N',
 #                   help='number of epochs to train (default: 10)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
   #                 help='enables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
 #                   help='random seed (default: 1)')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
 #                   help='how many batches to wait before logging training status')
#args = parser.parse_args()
cuda = torch.cuda.is_available()

#torch.manual_seed(1)

device = torch.device("cuda")# if cuda else "cpu")

#kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
def visualize_parameters(model):
    for n, p in model.named_parameters():
        if p.grad is None:
            print('{}\t{}\t{}'.format(n, p.data.norm(), None))
        else:
            print('{}\t{}\t{}'.format(n, p.data.norm(), p.grad.data.norm()))

class ListVAE(GaussianLatentVAE):
    def __init__(self, representation_size=0, lstsize=4, batch_size=128):
        #super(VAE, self).__init__()
        #self.save_init_params(locals())
        super().__init__(representation_size)
        np.random.seed(0)
        torch.manual_seed(1)
        self.fc1 = nn.Linear(lstsize, 64)
        self.fc21 = nn.Linear(64, 32)
        self.fc22 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, lstsize*10)
        self.batch_size = batch_size
        visualize_parameters(self)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        mid2 = self.fc4(h3).view(self.batch_size * 4, 10)
        return F.softmax(mid2, dim=-1)

    def forward(self, x):
        mu, logvar = self.encode(x)#.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def logprob(self, inputs, obs_distribution_params):
        pass



class MonetVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,
            attention_net,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
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
        self.attention_net = attention_net

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        # conv_output_size = deconv_args['deconv_input_width'] * \
        #                    deconv_args['deconv_input_height'] * \
        #                    deconv_args['deconv_input_channels']

        self.encoder = encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=self.imlength,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)

        self.fc1 = nn.Linear(self.encoder.output_size, representation_size)
        self.fc2 = nn.Linear(self.encoder.output_size, representation_size)

        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)

        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.decoder = decoder_class(
            **deconv_args,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            output_size=self.imlength,
            init_w=init_w,
            hidden_init=hidden_init,
            **deconv_kwargs)

        self.epoch = 0
        self.decoder_distribution = decoder_distribution

    def encode(self, input):
        h = self.encoder(input)
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return (mu, logvar)

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
        bs = input.shape[0]
        scope = from_numpy(np.ones((bs, 1, self.imsize, self.imsize)))
        log_scope = torch.log(scope)
        input = input.view(bs, 3, self.imsize, self.imsize)
        x_prob_losses = []
        kle_losses = []
        x_hats = []
        masks = []
        m_hat_logits = []
        log_masks = []
        for k in range(K):

            if k < K - 1:
                attention_input = torch.cat([input, log_scope], 1)
                attention_logits = self.attention_net(attention_input)
                log_mask = log_scope + F.logsigmoid(attention_logits)
            else:
                log_mask = log_scope

            if k == 0:
                sigma = 0.09
            else:
                sigma = 0.11
            mask = torch.exp(log_mask)

            vae_input = torch.cat([input, log_mask], 1)
            latent_distribution_params = self.encode(vae_input.view(bs, -1))
            latents = self.reparameterize(latent_distribution_params)
            x_hat, x_var_hat, m_hat_logit = self.decode(latents)

            x_hats.append(x_hat.detach())
            masks.append(mask.detach())
            m_hat_logits.append(torch.unsqueeze(m_hat_logit, 1))
            log_masks.append(log_mask)

            # 1. m outside log
            #x_prob = self.gaussian_log_prob(x_hat, input, from_numpy(np.array([sigma])))
            #x_prob_losses.append(mask * x_prob)

            # 2. m inside log
            x_prob = self.gaussian_prob(x_hat, input, from_numpy(np.array([sigma])))
            x_prob_losses.append(mask * x_prob)

            kle = self.kl_divergence(latent_distribution_params)
            kle_losses.append(kle)

            log_scope = log_scope + F.logsigmoid(-attention_logits) # log(s(1-attention)) = log(s) + log (1-attention)

        #import pdb; pdb.set_trace()
        all_m_hat_logits = torch.cat(m_hat_logits, 1)
        all_log_masks = torch.cat(log_masks, 1)
        mask_loss = F.softmax(all_m_hat_logits, 1) * (F.log_softmax(all_m_hat_logits, 1) - all_log_masks)
        mask_loss = mask_loss.sum() / input.shape[0]

        reconstruction = x_hats[0] * masks[0]
        for k in range(1, K):
            reconstruction += x_hats[k] * masks[k]

        return reconstruction, x_prob_losses, kle_losses, mask_loss, x_hats, masks


