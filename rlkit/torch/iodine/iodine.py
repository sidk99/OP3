import torch
import torch.utils.data
from rlkit.torch.iodine.physics_network import PhysicsNetwork, PhysicsNetworkMLP, PhysicsNetworkNoAttention, PhysicsNetworkAllAtOnce
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
        # K=7,
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
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

    ),
    physics_kwargs=dict(
        action_enc_size=32,
    ),
    schedule_kwargs=dict(
        train_T=5,
        test_T=5,
        seed_steps=4,
        schedule_type='single_step_physics'
    )
)

imsize64_large_iodine_architecture_multistep_physics = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=REPSIZE_128,
        input_channels=3,
        # decoder_distribution='gaussian_identity_variance',
        beta=1,
        # K=7, #7
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=80,
        input_height=80,
        input_channels=REPSIZE_128 + 2,

        kernel_sizes=[5, 5, 5, 5],
        n_channels=[64, 64, 64, 4], #Was 64
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

    ),
    physics_kwargs=dict(
        action_enc_size=32,
    ) #,
    # schedule_kwargs=dict(
    #     train_T=10,
    #     test_T=10,
    #     seed_steps=5,
    #     schedule_type='random_alternating'
    # )
)

imsize64_large_iodine_architecture_multistep_physics_BIG = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=128*4,
        input_channels=3,
        # decoder_distribution='gaussian_identity_variance',
        beta=1,
        # K=7, #7
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=80,
        input_height=80,
        input_channels=128*4 + 2,

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
        hidden_sizes=[128*4, 128*4],
        output_size=128*4,
        lstm_size=256*4,
        lstm_input_size=128*6*4,
        added_fc_input_size=0

    ),
    physics_kwargs=dict(
        action_enc_size=32,
    )
)

imsize64_large_iodine_architecture_multistep_physics_NoAttention = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=128,
        input_channels=3,
        # decoder_distribution='gaussian_identity_variance',
        beta=1,
        # K=7, #7
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=80,
        input_height=80,
        input_channels=128 + 2,

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
        lstm_input_size=128*6,
        added_fc_input_size=0

    ),
    physics_kwargs=dict(
        noattention=True,
        action_enc_size=32,
    )
)

imsize64_large_iodine_architecture_multistep_physics_AllAtOnce = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=128,
        input_channels=3,
        # decoder_distribution='gaussian_identity_variance',
        beta=1,
        # K=7, #7
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=80,
        input_height=80,
        input_channels=128 + 2,

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
        lstm_input_size=128*6,
        added_fc_input_size=0

    ),
    physics_kwargs=dict(
        allatonce=True,
        action_enc_size=32,
    )
)

imsize64_large_iodine_architecture_multistep_physics_MLP = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=128,
        input_channels=3,
        # decoder_distribution='gaussian_identity_variance',
        beta=1,
        # K=7, #7
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=80,
        input_height=80,
        input_channels=128 + 2,

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
        lstm_input_size=128*6,
        added_fc_input_size=0

    ),
    physics_kwargs=dict(
        mlp=True,
        action_enc_size=32,
    )
)

imsize64_small_iodine_architecture = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=REPSIZE_128,
        input_channels=3,
        beta=1,
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 4, #Note: This does not seem to be used in Broadcast
        input_width=64,
        input_height=64,
        input_channels=REPSIZE_128 + 2,

        kernel_sizes=[5, 5],
        n_channels=[64, 4],
        strides=[1, 1],
        paddings=[2, 2]
    ),
    deconv_kwargs=dict(
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    refine_args=dict(
        input_width=64,
        input_height=64,
        input_channels=17,
        paddings=[0, 0],
        kernel_sizes=[5, 5],
        n_channels=[64, 64],
        strides=[2, 2],
        hidden_sizes=[128, 128],
        output_size=REPSIZE_128,
        lstm_size=256,
        lstm_input_size=768,
        added_fc_input_size=0
    ),
    physics_kwargs=dict(
        action_enc_size=32,
    )
)

imsize64_medium_iodine_architecture = dict(
    vae_kwargs=dict(
        imsize=64,
        representation_size=REPSIZE_128,
        input_channels=3,
        beta=1,
        sigma=0.1,
    ),
    deconv_args=dict(
        hidden_sizes=[],
        output_size=64 * 64 * 3,
        input_width=64,
        input_height=64,
        input_channels=REPSIZE_128 + 2,

        kernel_sizes=[3, 3, 3, 3],
        n_channels=[64, 64, 64, 4],
        strides=[1, 1, 1 ,1],
        paddings=[1, 1, 1, 1]
    ),
    deconv_kwargs=dict(
        batch_norm_conv=False,
        batch_norm_fc=False,
    ),
    refine_args=dict(
        input_width=64,
        input_height=64,
        input_channels=17,
        paddings=[0, 0, 0],
        kernel_sizes=[3, 3, 3],
        n_channels=[64, 64, 64],
        strides=[1, 1, 1],
        hidden_sizes=[128, 128],
        output_size=REPSIZE_128,
        lstm_size=256,
        lstm_input_size=768,
        added_fc_input_size=0
    ),
    physics_kwargs=dict(
        action_enc_size=32,
    )
)

# schedule_parameters=dict(
#     train_T = 21,
#     test_T = 21,
#     seed_steps = 5,
#     schedule_type='random_alternating'
# )

#model, schedule_kwargs, K
def create_model(variant, action_dim):
    # if 'K' in variant.keys(): #New version
    K = variant['K']
    # else: #Old version
    #     K = variant['vae_kwargs']['K']
    print('K: {}'.format(K))
    model = variant['model']
    rep_size = model['vae_kwargs']['representation_size']

    decoder = BroadcastCNN(**model['deconv_args'], **model['deconv_kwargs'], hidden_activation=nn.ELU())
    refinement_net = RefinementNetwork(**model['refine_args'], hidden_activation=nn.ELU())

    if 'mlp' in model['physics_kwargs']:
        #Make another dictionary without the mlp key
        tmp = copy.deepcopy(model['physics_kwargs'])
        del tmp['mlp']
        physics_net = PhysicsNetworkMLP(K, rep_size, action_dim, **tmp)
    elif 'noattention' in model['physics_kwargs']:
        tmp = copy.deepcopy(model['physics_kwargs'])
        del tmp['noattention']
        physics_net = PhysicsNetworkNoAttention(K, rep_size, action_dim, **tmp)
    elif 'allatonce' in model['physics_kwargs']:
        tmp = copy.deepcopy(model['physics_kwargs'])
        del tmp['allatonce']
        physics_net = PhysicsNetworkNoAttention(K, rep_size, action_dim, **tmp)
    else:
        physics_net = PhysicsNetwork(K, rep_size, action_dim, **model['physics_kwargs'])

    if 'schedule_kwargs' not in variant:
        m = IodineVAE(
            **model['vae_kwargs'],
            K=K,
            decoder=decoder,
            refinement_net=refinement_net,
            physics_net=physics_net,
            action_dim=action_dim,
        )
    else:
        m = IodineVAE(
            **model['vae_kwargs'],
            **variant['schedule_kwargs'],
            K=K,
            decoder=decoder,
            refinement_net=refinement_net,
            physics_net=physics_net,
            action_dim=action_dim,
        )
    return m

def create_schedule(train, T, schedule_type, seed_steps, max_T=None):
    if schedule_type == 'single_step_physics':
        schedule = np.ones((T,))
        schedule[:seed_steps] = 0
    elif schedule_type == 'random_alternating':
        if train:
            schedule = np.random.randint(0, 2, (T,))
        else:
            schedule = np.ones((T,))
        schedule[:seed_steps] = 0
    elif schedule_type == 'multi_step_physics':
        schedule = np.ones((T, ))
        schedule[:seed_steps] = 0
    elif 'curriculum' in schedule_type:
        if train:
            max_multi_step = int(schedule_type[-1])
            # schedule = np.zeros((T,))
            rollout_len = np.random.randint(max_multi_step)+1
            schedule = np.zeros(seed_steps+rollout_len+1)
            schedule[seed_steps:seed_steps+rollout_len] = 1 #schedule looks like [0,0,0,0,1,1,1,0]
        else:
            max_multi_step = int(schedule_type[-1])
            schedule = np.zeros(seed_steps + max_multi_step+1)
            schedule[seed_steps:seed_steps + max_multi_step] = 1
    elif schedule_type == 'static_iodine':
        schedule = np.zeros((T,))
    elif schedule_type == 'rprp':
        schedule = np.zeros(seed_steps + (T-1)*2)
        schedule[seed_steps::2] = 1
    elif schedule_type == 'next_step':
        schedule = np.ones(T)*2
        return schedule
    elif schedule_type == 'occlusion':
        schedule = np.zeros(4+4)
        schedule[4::2] = 1  # 0,1,0,1,0
    else:
        raise Exception
    if max_T is not None: #Enforces that we have at most max_T-1 physics steps
        timestep_count = np.cumsum(schedule)
        schedule = np.where(timestep_count <= max_T-1, schedule, 0)
    # print(schedule)
    return schedule

####Get loss weight depending on schedule
def get_loss_weight(t, schedule, schedule_type):
    if schedule_type == 'single_step_physics':
        return t
    elif schedule_type == 'random_alternating':
        return t
    elif schedule_type == 'multi_step_physics':
        return t
    elif 'curriculum' in schedule_type:
        return t
    elif schedule_type == 'static_iodine':
        return t
    elif schedule_type == 'rprp':
        return t
    elif schedule_type == 'next_step':
        return t
    elif schedule_type == 'occlusion':
        return t
    else:
        raise Exception

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
            sigma=0.1,
            train_T=5,
            test_T=5,
            seed_steps=4,
            schedule_type='single_step_physics'

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
        # self.lstm_size = 256*4
        self.lstm_size = self.refinement_net.lstm_size
        # self.train_T = train_T
        self.test_T = test_T
        self.seed_steps = seed_steps
        self.schedule_type = schedule_type

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

        self.eval_mode = False


        
    def encode(self, input):
        pass

    def set_eval_mode(self, eval):
        self.eval_mode = eval

    def kl_divergence_softplus(self, latent_distribution_params):
        mu, softplus = latent_distribution_params
        stds = torch.sqrt(torch.log(1 + softplus.exp()))
        logvar = torch.log(torch.log(1 + softplus.exp()))
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - stds.pow(2), dim=1).sum()

    def decode(self, lambdas1, lambdas2, inputK, bs):
        #RV: inputK: (bs*K, ch, imsize, imsize)
        #RV: lambdas1, lambdas2: (bs*K, lstm_size)
        latents = self.rsample_softplus([lambdas1, lambdas2]) #lambdas1, lambdas2 are mu, softplus

        broadcast_ones = ptu.ones((latents.shape[0], latents.shape[1], self.decoder_imsize, self.decoder_imsize)).to(
            latents.device) #RV: (bs, lstm_size, decoder_imsize. decoder_imsize)
        decoded = self.decoder(latents, broadcast_ones) #RV: Uses broadcast decoding network, output (bs*K, 4, D, D)
        # print("decoded.shape: {}".format(decoded.shape))
        x_hat = decoded[:, :3] #RV: (bs*K, 3, D, D)
        m_hat_logits = decoded[:, 3] #RV: (bs*K, D, D), raw depth values

        m_hat_logit = m_hat_logits.view(bs, self.K, self.imsize, self.imsize) #RV: (bs, K, D, D)
        mask = F.softmax(m_hat_logit, dim=1)  # (bs, K, D, D)

        pixel_x_prob = self.gaussian_prob(x_hat, inputK, self.sigma).view(bs, self.K, self.imsize, self.imsize) #RV: Component p(x|h), (bs,K,D,D)
        pixel_likelihood = (mask * pixel_x_prob).sum(1)  # sum along K  #RV:sum over k of m_k*p_k, complete log likelihood
        log_likelihood = -torch.log(pixel_likelihood + 1e-12).sum() / bs #RV: This should be complete log likihood?

        kle = self.kl_divergence_softplus([lambdas1, lambdas2])
        kle_loss = self.beta * kle.sum() / bs #RV: KL loss
        loss = log_likelihood + kle_loss #RV: Total loss
        # pdb.set_trace()

        return x_hat, mask, m_hat_logits, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood

    def gaussian_prob(self, inputs, targets, sigma):
        ch = 3
        # (2pi) ^ ch = 248.05
        sigma = sigma.to(inputs.device)
        return torch.exp((-torch.pow(inputs - targets, 2).sum(1) / (ch * 2 * sigma ** 2))) / (
                    torch.sqrt(sigma ** (2 * ch)) * 248.05)

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
        T = masks.shape[1]
        m = masks[idx].permute(1, 0, 2, 3, 4).repeat(1, 1, 3, 1, 1)  # (K, T, ch, imsize, imsize)
        x = x_hats[idx].permute(1, 0, 2, 3, 4)
        rec = (m * x)
        full_rec = rec.sum(0, keepdim=True)

        comparison = torch.cat([ground_truth, full_rec, m, rec], 0).view(-1, 3, imsize, imsize)
        save_image(comparison.data.cpu(), logger.get_snapshot_dir() + '/goal_latents_%0.5f.png' % mse, nrow=T)


    #(3, D, D), (K, rep_size), (K, 3, D, D), (K, D, D)
    def refine(self, input, hidden_state, plot_latents=False):
        K = self.K
        bs = 8
        input = input.repeat(bs, 1, 1, 1).unsqueeze(1)

        T = 7 #Refine for 7 steps

        outputs = [[], [], [], [], []]
        # Run multiple times to get best one
        for i in range(6):
            x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = self._forward_dynamic_actions(
                input, None,
                schedule=np.zeros((T)))
            outputs[0].append(x_hats)
            outputs[1].append(masks)
            outputs[2].append(final_recon)
            outputs[3].append(lambdas[0].view(-1, K, self.representation_size))

        x_hats = torch.cat(outputs[0], 0)
        masks = torch.cat(outputs[1], 0)
        final_recon = torch.cat(outputs[2])
        lambdas = torch.cat(outputs[3], 0)
        

        lambda_recon = (x_hats * masks)
        recon = torch.clamp(final_recon, 0, 1)
        mse = torch.pow(final_recon - input[0], 2).mean(3).mean(2).mean(1)
        best_idx = torch.argmin(mse)
        if plot_latents:
            mses, best_idxs = mse.sort()
            for i in range(8):
                self.plot_latents(input[0].unsqueeze(0).repeat(1, T, 1, 1, 1), masks,
                                  x_hats, mse[best_idxs[i]], best_idxs[i])

        best_lambda = lambdas[best_idx]

        return recon[best_idx].data, best_lambda.data, lambda_recon[best_idx, -1].data, masks[best_idx, -1].data.squeeze()

    #input: should be tensor of shape (B, 3, D, D) or (B, T1, 3, D, D),
    #actions: None or (B, T2, A)
    def step(self, input, actions, plot_latents=False):
        if len(input.shape) == 4:
            input = input.unsqueeze(1) # RV: TODO: CHECK IF THIS WORKS

        K = self.K
        bs = input.shape[0]
        # imsize = self.imsize
        # input = input.unsqueeze(1).repeat(1, 9, 1, 1, 1) #RV: Why 9?

        # schedule = create_schedule(False, self.test_T, self.schedule_type, self.seed_steps) #RV: Returns schedule of 1's and 0's
        if actions is not None:
            #actions = actions.unsqueeze(1).repeat(1, 9, 1)
            self.test_T = self.seed_steps + actions.shape[1]
            schedule = np.ones((self.test_T,))
            schedule[:self.seed_steps] = 0
        else:
            schedule = np.zeros((self.seed_steps,))
            # schedule = create_schedule(False, self.test_T, self.schedule_type, self.seed_steps)  # RV: Returns schedule of 1's and 0's
        schedule = np.array([0,0,0,0,1])
        # print(schedule)


        #Note: self._forward_dynamic_actions require that the schedule physics steps have corresponding "true" image
        if sum(schedule) > input.shape[1] and input.shape[1] == 1:
            input = input.repeat(1, int(sum(schedule)+1), 1, 1, 1) #RV: TODO: This is very bad if T > 1
        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = self._forward_dynamic_actions(
            input=input, actions=actions, schedule=schedule)

        lambda_recon = (x_hats * masks)
        recon = torch.clamp(final_recon, 0, 1)
        if plot_latents:
            imsize = 64
            m = masks[0].permute(1, 0, 2, 3, 4).repeat(1, 1, 3, 1, 1)  # (K, T3, ch, imsize, imsize), T3 = seed_steps+T2
            x = x_hats[0].permute(1, 0, 2, 3, 4) # (
            rec = (m * x)
            full_rec = rec.sum(0, keepdim=True) #(1, 6, 3, imsize, imsize)

            input = input[:1].repeat(1, self.test_T, 1, 1, 1)

            comparison = torch.cat([input[0, :self.test_T].unsqueeze(0), full_rec, m, rec], 0).view(-1, 3, imsize, imsize)

            if isinstance(plot_latents, str):
                name = logger.get_snapshot_dir() + plot_latents
            else:
                name = logger.get_snapshot_dir() + '/test.png'

            save_image(comparison.data.cpu(), name, nrow=self.test_T)
            # save_image(comparison.data.cpu(), logger.get_snapshot_dir() + '/test.png', nrow=self.test_T)
        #  x_hats, 0, i)
        # pred_obs, obs_latents, obs_latents_recon

        return recon.data, lambdas[0].view(bs, K, -1).data, lambda_recon[:, -1].data

    #Batch method for step
    def step_batched(self, inputs, actions, bs=4):
        # Handle large obs in batches
        n_batches = int(np.ceil(inputs.shape[0] / float(bs)))
        outputs = [[], [], []]

        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, inputs.shape[0])
            if actions is not None:
                actions_batch = actions[start_idx:end_idx]
            else:
                actions_batch = None

            pred_obs, obs_latents, obs_latents_recon = self.step(inputs[start_idx:end_idx], actions_batch)
            outputs[0].append(pred_obs)
            outputs[1].append(obs_latents)
            outputs[2].append(obs_latents_recon)

        return torch.cat(outputs[0]), torch.cat(outputs[1]), torch.cat(outputs[2])

    #RV: Inputs: Information needed for IODINE refinement network (note much more information needed than RNEM)
    #RV: Outputs: Updates lambdas and hs
    def refine_lambdas(self, pixel_x_prob, pixel_likelihood, mask, m_hat_logit, loss, x_hat,
                       lambdas1, lambdas2, inputK, latents, h1, h2, tiled_k_shape, bs, add_fc_input=None):
        K = self.K
        lns = self.layer_norms
        posterior_mask = pixel_x_prob / (pixel_x_prob.sum(1, keepdim=True) + 1e-8)  # avoid divide by zero
        leave_out_ll = pixel_likelihood.unsqueeze(1) - mask * pixel_x_prob
        x_hat_grad, mask_grad, lambdas_grad_1, lambdas_grad_2 = \
            torch.autograd.grad(loss, [x_hat, mask] + [lambdas1, lambdas2],create_graph=not self.eval_mode, retain_graph=not self.eval_mode)

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

        # pdb.set_trace()

        lambdas1, lambdas2, h1, h2 = self.refinement_net(a, h1, h2,
                                                         extra_input=torch.cat([extra_input, lambdas1, lambdas2, latents], -1),
                                                         add_fc_input=add_fc_input)
        return lambdas1, lambdas2, h1, h2


    #RV: Input is (bs, T, ch, imsize, imsize), schedule is (T,): 0 for refinement and 1 for physics
    #    Runs refinement/dynamics on input accordingly into
    def _forward_dynamic_actions(self, input, actions, schedule, initial_lambdas=None):
        # input is (bs, T, ch, imsize, imsize)
        # schedule is (T,): 0 for refinement and 1 for physics
        K = self.K
        bs = input.shape[0]
        T = schedule.shape[0]

        # means and log_vars of latent
        if initial_lambdas is None:
            lambdas1 = self.lambdas1.unsqueeze(0).repeat(bs * K, 1) #(B*K, repsize)
            lambdas2 = self.lambdas2.unsqueeze(0).repeat(bs * K, 1) #(B*K, repsize)
        else:
            lambdas1 = self.lambdas1.unsqueeze(0).repeat(bs * K, 1)  # (B*K, repsize)
            lambdas1 += initial_lambdas[0].view(bs * K, -1) #(B*K, repsize)

            lambdas2 = self.lambdas2.unsqueeze(0).repeat(bs * K, 1)*0  # (B*K, repsize)
            lambdas2 += initial_lambdas[1].view(bs * K, -1) #(B*K, repsize)
        # initialize hidden state
        h1, h2 = self.initialize_hidden(bs * K) #RV: Each one is (bs * K, self.lstm_size)

        h1 = h1.to(input.device)
        h2 = h2.to(input.device)

        losses, x_hats, masks, kle_losses, log_prob_losses = [], [], [], [], []
        untiled_k_shape = (bs, K, -1, self.imsize, self.imsize)
        tiled_k_shape = (bs * K, -1, self.imsize, self.imsize)

        current_step = 0

        inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape) #RV: (bs*K, ch, imsize, imsize)
        x_hat, mask, m_hat_logit, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood = self.decode(
            lambdas1, lambdas2, inputK, bs) #RV: Returns sampled latents, decoded outputs, and computes the likelihood/loss
        losses.append(loss)
        kle_losses.append(kle_loss)
        log_prob_losses.append(log_likelihood)

        for t in range(1, T+1):
            # if lambdas1.shape[0] % self.K != 0:
            #     raise ValueError("Incorrect lambdas1 shape: {}".format(lambdas1.shape))
            if schedule[t - 1] == 0: # Refine
                inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape) #RV: (bs*K, ch, imsize, imsize)
                lambdas1, lambdas2, h1, h2 = self.refine_lambdas(pixel_x_prob, pixel_likelihood, mask, m_hat_logit,
                                                                 loss, x_hat, lambdas1, lambdas2, inputK, latents, h1, h2,
                                                                 tiled_k_shape, bs) #RV: Update lambdas and h's using info
                # if not applied_action: # Do physics on static scene if haven't applied action yet
                #     lambdas1, _ = self.physics_net(lambdas1, lambdas2, None)
            elif schedule[t-1] == 1: # Physics
                current_step += 1
                if actions is not None:
                    actionsK = actions[:, current_step - 1].unsqueeze(1).repeat(1, K, 1).view(bs * K, -1)
                else:
                    actionsK = None

                if current_step >= input.shape[1] - 1:
                    current_step = input.shape[1] - 1
                inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)
                lambdas1, _ = self.physics_net(lambdas1, lambdas2, actionsK)
                # loss_w = t #RV modification
            elif schedule[t-1] == 2: #Next step refinement
                current_step += 1
                if actions is not None:
                    actionsK = actions[:, current_step - 1].unsqueeze(1).repeat(1, K, 1).view(bs * K, -1)
                    if actionsK.shape[-1] == 6 and self.refinement_net.added_fc_input_size == 4:
                        actionsK = actionsK[:, torch.LongTensor([0, 1, 3, 4])]
                else:
                    actionsK = None
                inputK = input[:, current_step].unsqueeze(1).repeat(1, K, 1, 1, 1).view(tiled_k_shape)
                lambdas1, lambdas2, h1, h2 = self.refine_lambdas(pixel_x_prob, pixel_likelihood, mask, m_hat_logit,
                                                                 loss, x_hat, lambdas1, lambdas2, inputK, latents, h1,
                                                                 h2, tiled_k_shape, bs, add_fc_input=actionsK)  # RV: Update lambdas and h's using info
            else:
                raise ValueError("Invalid schedule value: {}".format(schedule[t-1]))


            loss_w = get_loss_weight(t, schedule, self.schedule_type) + 1

            # Decode and get loss
            x_hat, mask, m_hat_logit, latents, pixel_x_prob, pixel_likelihood, kle_loss, loss, log_likelihood = \
                self.decode(lambdas1, lambdas2, inputK, bs)

            x_hats.append(x_hat.data)
            masks.append(mask.data)
            losses.append(loss * loss_w)

            kle_losses.append(kle_loss * loss_w)
            log_prob_losses.append(log_likelihood * loss_w)


        total_loss = sum(losses) / ((T+1)*(T+2)/2)
        kle_loss = sum(kle_losses) / ((T+1)*(T+2)/2)
        log_likelihood = sum(log_prob_losses) / ((T+1)*(T+2)/2)

        final_recon = (mask.unsqueeze(2) * x_hat.view(untiled_k_shape)).sum(1)
        mse = torch.pow(final_recon - input[:, -1], 2).mean()

        all_x_hats = torch.stack([x.view(untiled_k_shape) for x in x_hats], 1)  # (bs, T, K, 3, imsize, imsize)
        all_masks = torch.stack([x.view(untiled_k_shape) for x in masks], 1)  # # (bs, T, K, 1, imsize, imsize)
        return all_x_hats.data, all_masks.data, total_loss, kle_loss.data, \
               log_likelihood.data, mse, final_recon.data, [lambdas1.data, lambdas2.data]


    #input:tensor of shape (B, T1, 3, D, D), actions: (B, T2, A) where T2 < T1
    # def special_forward(self, obs, actions):

    #High level: Need a function that takes in a sequence of actions and images and returns hidden state (dynamic seed steps)
    #Then using hidden state, run dynamics (action selection) multiple times and return images/lambdas accordingly
    #Once we have choosen an action, run dynamics to get new hidden state and repeat process

    #Input: Both are tensors, obs: (B, T, 3, D, D), actions: (B, T-1, A) or None
    def get_hidden_state(self, obs, actions, plot_image_file_name=None):
        bs, T = obs.shape[0], obs.shape[1]
        seed_steps = 4
        num_refine_per_phys = 4
        schedule = np.zeros(seed_steps + (T - 1) * num_refine_per_phys)  # len(schedule) = T2
        schedule[seed_steps::num_refine_per_phys] = 1  # [0,0,0,0,1,0,1,0,1,0] if num_refine_per_phys=2 for example

        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = self._forward_dynamic_actions(obs, actions, schedule)

        object_recons = x_hats * masks  # (bs, T2, K, 3, D, D)
        # object_recons = object_recons[:, seed_steps - 1::num_refine_per_phys]  # (bs, T, K, 3, D, D)
        # final_recons = object_recons.sum(2, keepdim=True)  # (bs, T, 1, 3, D, D)

        if plot_image_file_name is not None:
            true_obs = obs[:, np.cumsum(schedule)].unsqueeze(2) #(B, T2, 3, D, D) -> (B, T2, 1, 3, D, D)
            final_recons = object_recons.sum(2, keepdim=True)  # (bs, T2, 1, 3, D, D)
            all_object_recons = torch.cat([true_obs, final_recons, object_recons], dim=2)  # (bs=1, T2, K+2, 3, D, D)
            all_object_recons = all_object_recons.squeeze(0) #(T2, K+2, 3, D, D)
            all_object_recons = all_object_recons.permute(1, 0, 2, 3, 4).contiguous()  #(K+2, T2, 3, D, D)
            all_object_recons = all_object_recons.view(-1, *all_object_recons.shape[-3:])
            save_image(all_object_recons, filename=plot_image_file_name, nrow=final_recons.shape[1])

        object_recons = object_recons[:, seed_steps - 1::num_refine_per_phys]  # (bs, T, K, 3, D, D)
        final_recons = object_recons.sum(2, keepdim=True)  # (bs, T, 1, 3, D, D)

        return [lambdas[0].view(bs, self.K, -1).detach(), lambdas[1].view(bs, self.K, -1).detach()], final_recons

    # Input: Both are tensors, lambdas: Tuple of two, each element is (B, K, repsize), actions: (B, T, A)
    # Output: full_image: (B, 3, D, D),  lambdas: Tuple of two (B, K, repsize),  sub_images:(B, K, 3, D, D)
    def run_dynamics(self, lambdas, actions):
        bs, k, rep_size = lambdas[0].shape
        T = actions.shape[1]
        lambdas1 = lambdas[0].view(-1, rep_size) #(B*K, rep_size)
        lambdas2 = lambdas[1].view(-1, rep_size) #(B*K, rep_size)

        for i in range(T):
            cur_actions = actions[:, i] #(B, A)
            cur_actions = cur_actions.unsqueeze(1).repeat(1, k, 1).view(bs*k, -1) #(B, A) -> (B, 1, A) -> (B, K, A) -> (B*K, A)
            lambdas1, _ = self.physics_net(lambdas1, lambdas2, cur_actions)

        lambdas1 = lambdas1.view(bs, k, rep_size) #(B, K, rep_size)
        lambdas2 = lambdas2.view(bs, k, rep_size) #(B, K, rep_size)
        x_hats, masks = self.just_decode([lambdas1, lambdas2]) #x_hats: (bs, K, 3, D, D), masks: (bs, K, D, D)

        sub_images = x_hats * masks.unsqueeze(2) #(bs, K, 3, D, D)
        full_image = sub_images.sum(1) #(bs, 3, D, D)

        return full_image, [lambdas1.detach(), lambdas2.detach()], sub_images

    #Input: lambdas: Tuple of two, each element is (N, K, repsize), actions: (N, T, A)
    # Output: full_image: (N, 3, D, D),  lambdas: Tuple of two (N, K, repsize),  sub_images:(N, K, 3, D, D)
    def batch_run_dynamics(self, lambdas, actions, bs=4):
        # Handle large obs in batches
        N = actions.shape[0]
        n_batches = int(np.ceil(N / float(bs)))
        outputs = [[], [], [], []]

        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, N)
            if actions is not None:
                actions_batch = actions[start_idx:end_idx]
            else:
                actions_batch = None
            lambdas_batch = [lambdas[0][start_idx:end_idx], lambdas[1][start_idx:end_idx]]

            pred_obs, obs_latents, obs_latents_recon = self.run_dynamics(lambdas_batch, actions_batch)
            outputs[0].append(pred_obs)
            outputs[1].append(obs_latents[0])
            outputs[2].append(obs_latents[1])
            outputs[3].append(obs_latents_recon)

        return torch.cat(outputs[0]), [torch.cat(outputs[1]), torch.cat(outputs[2])], torch.cat(outputs[3])

    #Input: lambdas: Tuple of two, each element is (B, K, repsize)
    # By default we want to return probability mask and not depth values
    def just_decode(self, lambdas, give_depth_instead_of_mask=False):
        bs, k, rep_size = lambdas[0].shape

        lambdas1 = lambdas[0].view(bs*k, rep_size) #(B*K, repsize)
        lambdas2 = lambdas[1].view(bs*k, rep_size) #(B*K, repsize)
        latents = self.rsample_softplus([lambdas1, lambdas2])  # lambdas1, lambdas2 are mu, softplus

        broadcast_ones = ptu.ones((latents.shape[0], latents.shape[1], self.decoder_imsize, self.decoder_imsize)).to(
            latents.device)  # RV: (bs, lstm_size, decoder_imsize. decoder_imsize)
        decoded = self.decoder(latents, broadcast_ones)  # RV: Uses broadcast decoding network, output (bs*K, 4, D, D)
        # print("decoded.shape: {}".format(decoded.shape))
        x_hat = decoded[:, :3]  # RV: (bs*K, 3, D, D)
        m_hat_logits = decoded[:, 3]  # RV: (bs*K, 1, D, D), raw depth values

        x_hat = x_hat.view(bs, k, 3, self.imsize, self.imsize)  # (bs, K, 3, D, D)
        m_hat_logits = m_hat_logits.view(bs, k, self.imsize, self.imsize)  # RV: (bs, K, D, D)

        if not give_depth_instead_of_mask: #If false, give mask
            mask = F.softmax(m_hat_logits, dim=1)  # (bs, K, D, D)
        else: #Give depth values
            mask = m_hat_logits # (bs, K, D, D)
        return x_hat.detach(), mask.detach()

    # cur_lambdas: Tuple of size two, each of (B, K, repsize), 
    # obs: (B, T, 3, D, D), actions: (B, T, A)
    #Note actions is (_, T, _), so we lead with a physics step
    def update_lambdas_with_obs_action(self, cur_lambdas, obs, actions):
        self.eval_mode = True
        first_action = actions[:, :1] #(B, 1, A)
        _, new_lambdas, _ = self.run_dynamics(cur_lambdas, first_action) #Predicts first image

        bs, T = obs.shape[0], obs.shape[1]
        num_refine_per_phys = 1
        schedule = np.zeros(T * (num_refine_per_phys+1))
        schedule[::(num_refine_per_phys+1)] = 1  # [1,0,1,0] if num_refine_per_phys=1 & T=2 for example
        schedule = schedule[1:] #As we already did the first physics step

        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = \
            self._forward_dynamic_actions(obs, actions[:, 1:], schedule, initial_lambdas=new_lambdas)

        lambdas[0] = lambdas[0].view(bs, self.K, -1).detach()
        lambdas[1] = lambdas[1].view(bs, self.K, -1).detach()
        return lambdas




