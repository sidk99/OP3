import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pytorch_utils as ptu
from pytorch_utils import apply_act, apply_LN_conv, init_weights, LayerNorm, orthog_init

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HIDDEN_SIZE = 500

#######################################Start building the encoder networks##################################
class MuSigmaEncoder(nn.Module):
    def __init__(self, color=False):
        super(MuSigmaEncoder, self).__init__()
        self.in_ch = 1
        if color:
            self.in_ch = 3

        self.conv1 = nn.Conv2d(self.in_ch, 32, (4,4), (2,2), padding=(1, 1))
        self.ln_c1 = LayerNorm([32])
        self.conv2 = nn.Conv2d(32, 32, (4,4), (2,2), padding=(1, 1))

        self.ln_c2 = LayerNorm([32])

        self.conv3 = nn.Conv2d(32, 64, (4,4), (2,2), padding=(1, 1))
        self.ln_c3 = LayerNorm([64])

        self.fc1 = nn.Linear(8*8*64, HIDDEN_SIZE)
        self.ln_f1 = LayerNorm(HIDDEN_SIZE)

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, self.in_ch, 64, 64) #(B, K, 64, 64, 1) -> (B*K, 1, 64, 64)
        x = apply_act(apply_LN_conv(self.conv1(x), self.ln_c1), 'elu') #(B*K, 1, 64, 64) -> (B*K, 16, 32, 32)
        x = apply_act(apply_LN_conv(self.conv2(x), self.ln_c2), 'elu') #(B*K, 16, 32, 32) -> (B*K, 32, 16, 16)
        x = apply_act(apply_LN_conv(self.conv3(x), self.ln_c3), 'elu') #(B*K, 32, 16, 16) -> (B*K, 64, 8, 8)

        x = x.view((x.size()[0], -1)) #(B*K, 64, 8, 8) -> (B*K, 64*8*8=4096)
        # x = apply_act(self.ln_f1(self.fc1(x)), 'elu') #(B*K, 64*8*8=4096) -> (B*K, 512)
        x = self.ln_f1(self.fc1(x))
        return x

class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (4, 4), (2, 2), padding=(1, 1))
        self.ln_c1 = LayerNorm([32])

        self.conv2 = nn.Conv2d(32, 32, (4, 4), (2, 2), padding=(1, 1))
        self.ln_c2 = LayerNorm([32])

        self.conv3 = nn.Conv2d(32, 64, (4, 4), (2, 2), padding=(1, 1))
        self.ln_c3 = LayerNorm([64])

        self.fc1 = nn.Linear(8 * 8 * 64, HIDDEN_SIZE)
        self.ln_f1 = LayerNorm(HIDDEN_SIZE)

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)  # (B, K, 64, 64, 1) -> (B*K, 1, 64, 64)
        x = apply_act(apply_LN_conv(self.conv1(x), self.ln_c1), 'elu')  # (B*K, 1, 64, 64) -> (B*K, 16, 32, 32)
        x = apply_act(apply_LN_conv(self.conv2(x), self.ln_c2), 'elu')  # (B*K, 16, 32, 32) -> (B*K, 32, 16, 16)
        x = apply_act(apply_LN_conv(self.conv3(x), self.ln_c3), 'elu')  # (B*K, 32, 16, 16) -> (B*K, 64, 8, 8)

        x = x.view((x.size()[0], -1))  # (B*K, 64, 8, 8) -> (B*K, 64*8*8=4096)
        # x = apply_act(self.ln_f1(self.fc1(x)), 'elu')  # (B*K, 64*8*8=4096) -> (B*K, 512)
        x = self.ln_f1(self.fc1(x))
        return x
#######################################End building the encoder networks##################################


###################################Start building the recurrent interaction network#############################
class TheRecurrentNet(nn.Module):
    def __init__(self, has_action, k, h, action_size, device):
        super(TheRecurrentNet, self).__init__()
        self.has_action = has_action
        self.h = h
        self.k = k
        self.device = device
        self.action_size = action_size

        ####Start physics####
        self.fc1 = nn.Linear(h, h) #Encoder
        self.ln_f1 = LayerNorm(h) #LN on output

        self.fc2 = nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE) #Core: (Focus, context)
        self.ln_f2 = LayerNorm(HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) #Context
        self.ln_f3 = LayerNorm(HIDDEN_SIZE)

        self.fc4 = nn.Linear(HIDDEN_SIZE, 100) #Attention
        self.ln_f4 = LayerNorm(100)
        self.fc5 = nn.Linear(100, 1) #h + h2 + m

        self.fc6 = nn.Linear(h+h, h) #Recurrent update
        self.ln_f6 = LayerNorm(h)
        ####End physics####

        ####Start actions####
        self.ac_fc1 = nn.Linear(action_size, h) #Action encoder
        self.ac_ln_f1 = LayerNorm(h)

        self.ac_fc2 = nn.Linear(h+h, h) #Effect
        self.ac_ln_f2 = nn.Linear(h, h)
        self.ac_fc3 = nn.Linear(h+h, 1) #Attention

        ####End actions####
        self.apply(init_weights)


    def action_forward(self, state, actions):
        #state: (batch_size * k, h)
        #actions: (batch_size, A), where A is size of action space
        bk, m = state.size()
        b = bk // self.k
        k = self.k
        h_state = state.size()[1]
        # state = state.view([b, k, h_state])
        if actions is None:
            return state

        actions = self.ac_ln_f1(self.ac_fc1(actions)) #(batch_size, hA)
        h_actions = actions.size()[1]
        actions = actions.view([b, 1, h_actions]) #(batch_size, 1, hA)
        actions = actions.repeat([1, k, 1]) #(batch_size, k, hA)
        actions = actions.view([b*k, h_actions])

        concat = torch.cat([state, actions], dim=1)  # (b x k, h + hA)
        attention = apply_act(self.ac_fc3(concat), 'sigmoid')
        effect = self.ac_ln_f2(self.ac_fc2(concat))

        net_effect = effect * attention
        return net_effect


    def forward(self, state, actions):
        bk, m = state.size() #(batch_size * k, h)
        b = bk // self.k
        k = self.k
        #Action: (b, A)

        if self.has_action:
            state = self.action_forward(state, actions)

        state1 = apply_act(self.ln_f1(self.fc1(state)), 'relu')

        if k == 1:
            tmp = torch.zeros((b*k, self.h), device=self.device)
            total = torch.cat([state1, tmp], dim=1) # (b x k, h + m)
            new_state = apply_act(self.ln_f6(self.fc6(total)), 'sigmoid')
            return new_state, new_state

        # Reshape theta to get used for context
        h1 = state1.size()[1]
        state1r = state1.view([b, k, h1])

        # Reshape theta to be used for focus
        state1rr = state1r.view([b, k, 1, h1])

        # Create focus: tile state1rr k-1 times
        # fs = state1rr.repeat([1, 1, k - 1, 1])  # (b, k, k-1, h1)
        num_context = k - 1
        # if self.has_action:
        #     num_context = k
        #     actions = apply_act(self.ac_fc1(actions), 'relu')

        fs = state1rr.repeat([1, 1, num_context, 1]) #k-1 other embeddings, +1 for action

        # Create context
        state1rl = torch.unbind(state1r, dim=1)

        csu = []
        for i in range(k):  # RV: Going through k thetas
            selector = [j for j in range(k) if j != i]  # RV: Getting indices not equal to k
            c = list(np.take(state1rl, selector))  # list of length k-1 of (b, h1) #RV: Selecting corresponding thetas
            # if self.has_action:
            #     c.append(actions) #Adding actions
            c = torch.stack(c, dim=1)  # (b, num_context, h1) #RV: Stacking list into single tensor
            csu.append(c)

        cs = torch.stack(csu, dim=1)  # (b, k, num_context, h1) with actions  #RV: Stacks list of tensor into single tensor

        # Reshape focus and context
        # you will process the k-1 instances through the same network anyways
        # fsr = fs.view([b * k * (k - 1), h1])  # (b x k x k-1, h1)
        # csr = cs.view([b * k * (k - 1), h1])  # (b x k x k-1, h1)
        # RV: Above reshapes focus and context
        fsr = fs.view([b * k * num_context, h1])  # (b x k x num_context, h1)
        csr = cs.view([b * k * num_context, h1])  # (b x k x num_context, h1)

        # Concatenate focus and context
        concat = torch.cat([fsr, csr], dim=1)  # (b x k x num_context, 2h1) with actions
        # print('concat: {}'.format(concat.size()))
        # RV: Above concatenates focus and context, this acts as the pairwise interactions

        # NPE core
        core_out = apply_act(self.ln_f2(self.fc2(concat)), 'relu')

        # Context branch: produces context
        context = apply_act(self.ln_f3(self.fc3(core_out)), 'relu')

        h2 = HIDDEN_SIZE
        contextr = context.view([b*k, num_context, h2])

        # Attention branch
        attention = apply_act(self.ln_f4(self.fc4(core_out)), 'tanh')
        attention = apply_act(self.fc5(attention), 'sigmoid')
        attentionr = attention.view([b*k, num_context, 1])
        effectrsum = torch.sum(contextr * attentionr, dim=1)

        total = torch.cat([state1, effectrsum], dim=1)  # (b x k, h + h2 + m)
        # new_state = apply_act(self.ln_f6(self.fc6(total)), 'sigmoid')
        # h_new = apply_act(self.fc6(total), 'sigmoid')
        h_new = self.ln_f6(self.fc6(total))

        return h_new
###################################End building the recurrent interaction network#############################


class MuDecoderNet(nn.Module):
    def __init__(self, pixel_dist, shared_decoder, color=False):
        self.pixel_dist = pixel_dist
        self.shared_decoder = shared_decoder
        self.out_ch = 1
        if color:
            self.out_ch = 3
        if shared_decoder:
            self.out_ch += 1

        super(MuDecoderNet, self).__init__()
        self.fc1 = nn.Linear(HIDDEN_SIZE, 512)
        self.ln_f1 = LayerNorm(512)

        self.fc2 = nn.Linear(512, 8*8*64)
        self.ln_f2 = LayerNorm(8*8*64)

        self.reshape_size1 = (2 * 8 + 1, 2 * 8 + 1)
        self.conv1 = nn.Conv2d(64, 32, (4, 4), stride=(1, 1), padding=(1, 1))
        self.ln_c1 = LayerNorm([32])

        self.reshape_size2 = (2 * 16 + 1, 2 * 16 + 1)

        self.conv2 = nn.Conv2d(32, 16, (4, 4), stride=(1, 1), padding=(1, 1))
        self.ln_c2 = LayerNorm([16])

        self.reshape_size3 = (2 * 32 + 1, 2 * 32 + 1)
        self.conv3 = nn.Conv2d(16, self.out_ch, (4, 4), stride=(1, 1), padding=(1, 1))

        self.apply(init_weights)

    def forward(self, x):
        x = apply_act(self.ln_f1(self.fc1(x)), 'relu')
        x = apply_act(self.ln_f2(self.fc2(x)), 'relu')
        x = x.view((x.size()[0], 64, 8, 8))

        x = nn.functional.interpolate(x, self.reshape_size1, mode="bilinear")
        x = apply_act(apply_LN_conv(self.conv1(x), self.ln_c1), 'relu')

        x = nn.functional.interpolate(x, self.reshape_size2, mode="bilinear")
        x = apply_act(apply_LN_conv(self.conv2(x), self.ln_c2), 'relu')

        x = nn.functional.interpolate(x, self.reshape_size3, mode="bilinear")
        x = self.conv3(x)
        x = x.permute(0, 2, 3, 1)

        if self.shared_decoder:
            mu = x[:, :, :, :x.shape[-1]-1]
            depth = torch.unsqueeze(x[:, :, :, -1], -1)
            depth = apply_act(depth, 'relu')

            if self.pixel_dist == 'bernoulli':
                mu = apply_act(mu, 'sigmoid')
            return mu, depth
        else:
            if self.pixel_dist == 'bernoulli':
                x = apply_act(x, 'sigmoid')
            return x


class DepthDecoderNet(nn.Module):
    def __init__(self):
        super(DepthDecoderNet, self).__init__()
        self.fc1 = nn.Linear(HIDDEN_SIZE, 512)
        self.ln_f1 = LayerNorm(512)

        self.fc2 = nn.Linear(512, 8*8*64)
        self.ln_f2 = LayerNorm(8*8*64)

        self.reshape_size1 = (2 * 8 + 1, 2 * 8 + 1)
        self.conv1 = nn.Conv2d(64, 32, (4, 4), stride=(1, 1), padding=(1, 1))
        self.ln_c1 = LayerNorm([32])

        self.reshape_size2 = (2 * 16 + 1, 2 * 16 + 1)
        self.conv2 = nn.Conv2d(32, 16, (4, 4), stride=(1, 1), padding=(1, 1))
        self.ln_c2 = LayerNorm([16])

        self.reshape_size3 = (2 * 32 + 1, 2 * 32 + 1)
        self.conv3 = nn.Conv2d(16, 1, (4, 4), stride=(1, 1), padding=(1, 1))

        self.apply(init_weights)

    def forward(self, x):
        x = apply_act(self.ln_f1(self.fc1(x)), 'relu')
        x = apply_act(self.ln_f2(self.fc2(x)), 'relu')
        x = x.view((x.size()[0], 64, 8, 8))

        x = nn.functional.interpolate(x, self.reshape_size1, mode="bilinear")
        x = apply_act(apply_LN_conv(self.conv1(x), self.ln_c1), 'relu')

        x = nn.functional.interpolate(x, self.reshape_size2, mode="bilinear")
        x = apply_act(apply_LN_conv(self.conv2(x), self.ln_c2), 'relu')

        x = nn.functional.interpolate(x, self.reshape_size3, mode="bilinear")
        x = apply_act(self.conv3(x), 'relu')
        # x = self.conv3(x)

        return x.permute(0, 2, 3, 1)


class Entire_RNEM(nn.Module):
    def __init__(self, has_action, k, device, training_info, color):
        super(Entire_RNEM, self).__init__()
        self.has_action = has_action
        self.device = device
        self.mu_sigma_enc = MuSigmaEncoder(color=color).to(device)
        self.depth_enc = DepthEncoder().to(device)
        self.recurrent = TheRecurrentNet(has_action=has_action, k=k, action_size=training_info['action_size'], h=HIDDEN_SIZE,
                                         device=device).to(device)
        self.mu_dec = MuDecoderNet(training_info["pixel_dist"], training_info["shared_mu_depth_decoder"], color=color).to(device)
        if not training_info["shared_mu_depth_decoder"]:
            self.depth_dec = DepthDecoderNet().to(device)

        self.training_info = training_info

    def forward(self, x, h_old, d_preds, gamma, actions):
        batch_size, k = gamma.size()[0], gamma.size()[1]
        #x of shape (B, K, W, H, C)
        #depth_probs, gamma of shape (B, K, W, H, 1)
        #h_old of shape (B*K, Hidden)

        h_mstep = self.run_improvement(x, h_old, d_preds, gamma)
        h_new = self.run_phyics(h_mstep, actions)
        mu_preds, depth_preds = self.decode(h_new)
        return mu_preds, depth_preds, h_new

    def forward_rollout(self, h_old, actions):
        h_new = self.run_phyics(h_old, actions)
        mu_preds, depth_preds = self.decode(h_new)
        return h_new, mu_preds, depth_preds

    def decode(self, h_cur):
        if self.training_info["shared_mu_depth_decoder"]:
            mu_preds, depth_preds = self.mu_dec(h_cur)
        else:
            mu_preds = self.mu_dec(h_cur)
            depth_preds = self.depth_dec(h_cur)  # (B*K, W, H, 1)
        return mu_preds, depth_preds

    def run_phyics(self, h_old, actions):
        h_new = self.recurrent(h_old, actions)  # (B*K, Hi)
        return h_new

    def run_improvement(self, x, h_old, d_preds, gamma):
        # x:(B, K, W, H, C), depth_probs & gamma:(B, K, W, H, 1), h_old:(B*K, Hidden)
        depth_probs = torch.nn.functional.softmax(d_preds, dim=1)

        tmp1 = self.depth_enc(gamma - depth_probs)  # depth_probs - gamma
        tmp2 = self.mu_sigma_enc(x * gamma)  # dQ_du (B*K, Hi)
        h_mstep = h_old + tmp1 + tmp2
        return h_mstep

    def visualize(self):
        # ptu.visualize_parameters(self.mu_sigma_enc, "Mu Encoder")
        # ptu.visualize_parameters(self.depth_enc, "Depth Encoder")
        # ptu.visualize_parameters(self.mu_dec, "Mu Decoder")
        ptu.visualize_parameters(self.recurrent, "Recurrent")
        # if not self.training_info["shared_mu_depth_decoder"]:
        #     ptu.visualize_parameters(self.depth_dec, "Depth Decoder")



class Entire_NEM(nn.Module):
    def __init__(self, rnem_cell, input_shape, pred_init, device, training_info):
        super(Entire_NEM, self).__init__()
        self.device = device
        self.rnem_cell = rnem_cell
        self.input_shape = input_shape #(W, H, C)
        self.pred_init = pred_init
        self.pixel_dist = training_info["pixel_dist"]
        self.e_sigma = training_info["e_sigma"]
        self.training_info = training_info

        self.rollout_prob = 0
        self.static_prob = training_info["static_prob"]
        self.improve_prob = training_info["improve_prob"]

    def init_state(self, batch_size, k, dtype):
        # inner RNN hidden state init
        h = torch.zeros((batch_size*k, 500), dtype=dtype, device=self.device) #TODO: Check shape!

        # initial prediction (B, K, W, H, C)
        pred_shape = [batch_size, k] + list(self.input_shape)
        pred = torch.ones(pred_shape, dtype=dtype, device=self.device) * self.pred_init

        gamma_shape = list(self.input_shape[:-1]) + [1] #RV: (W, H, 1)
        gamma_shape = [batch_size, k] + gamma_shape #(B, K, W, H, 1)

        gamma = torch.distributions.normal.Normal(0, 1).sample(torch.Size(gamma_shape)).to(self.device)
        gamma = torch.abs(gamma) + 1e-6
        gamma /= torch.sum(gamma, dim=1, keepdim=True) #(B, K, W, H, 1)

        d_preds = torch.abs(torch.distributions.normal.Normal(0, 1).sample(torch.Size(gamma_shape)).to(self.device))
        return h, pred, d_preds, gamma.detach()

    def e_step(self, preds, targets, depth_preds):
        # Initial part computes pixelwise loss of predictions in respect to target
        # predictions: (B, K, W, H, C), data: (B, 1, W, H, C), prob: (B, K, W, H, 1)

        if targets is None:
            return torch.nn.functional.softmax(depth_preds, dim=1).detach()

        if self.pixel_dist == "bernoulli":
            mu = preds
            prob = targets * mu + (1 - targets) * (1 - mu)  # RV: Note data is binary
        elif self.pixel_dist == "gaussian":
            mu, sigma = preds, self.e_sigma
            prob = ((1 / torch.FloatTensor([np.sqrt((2 * np.pi * sigma ** 2))]).to(self.device))
                    * torch.exp(torch.sum(-torch.pow(targets - mu, 2), dim=-1, keepdim=True) / (2 * sigma ** 2)))
            #prob = torch.sum(prob, dim=-1, keepdim=True) # TODO Check if right for color
        else:
            raise ValueError('Unknown distribution_type: "{}"'.format(self.pixel_dist))

        if self.training_info["include_depth"]:
            depth_probs = torch.nn.functional.softmax(depth_preds, dim=1)
            gamma = depth_probs * prob + 1e-6  # RV: Computing posterior probability
        else:
            gamma = prob + 1e-6

        gamma = gamma/torch.sum(gamma, dim=1, keepdim=True)
        return gamma.detach()


    # def forward(self, inputs, state, actions):
    #     input_data, target_data = inputs  # RV: input_data=t'th noisy data, target_data=t+1'th true data
    #     # Both of shape (B, 1, W, H, C)
    #     batch_size = input_data.size()[0]
    #     h_old, preds_old, d_preds_old, gamma_old = state
    #
    #     deltas = input_data - preds_old  # RV: delta = input_data - preds_old, implicit broadcasting
    #     if self.pixel_dist == "gaussian":
    #         deltas = deltas / (self.training_info["e_sigma"] ** 2)
    #     elif self.pixel_dist == "bernoulli":
    #         deltas = deltas / ((preds_old * (1 - preds_old)) + 1e-6)
    #
    #     # RV: Note preds_old=(B, K, W, H, C), input_data = (B, 1, W, H, C), deltas=(B, K, W, H, C)
    #     mu_preds, d_preds, h_new = self.rnem_cell(deltas, h_old, d_preds_old, gamma_old, actions)
    #
    #     # MC: reshape from (B*K, W, H, C) to (B, K, W, H, C)
    #     mu_preds = mu_preds.view([batch_size, -1] + list(mu_preds.size()[1:]))
    #     d_preds = d_preds.view([batch_size, -1] + list(d_preds.size()[1:]))
    #
    #     gamma = self.e_step(mu_preds, target_data, d_preds)
    #
    #     return h_new, mu_preds, d_preds, gamma.detach()

    # def forward_rollout(self, h_old, actions, target_shape):
    #     #target_shape = (B, ?, W, H, C)
    #     #h_old of size B*K
    #     h_new, mu_preds, depth_preds = self.rnem_cell.forward_rollout(h_old, actions)
    #
    #     mu_preds = mu_preds.view([target_shape[0], -1] + target_shape[2:])
    #     depth_preds = depth_preds.view([target_shape[0], -1] + target_shape[2:-1] + [1])
    #     return h_new, mu_preds, depth_preds, torch.nn.functional.softmax(depth_preds, dim=1).detach()

    # def improve(self, input_image, state):
    #     batch_size = input_image.size()[0]
    #     h_old, preds_old, d_preds_old, gamma_old = state
    #     deltas = input_image - preds_old
    #     if self.pixel_dist == "gaussian":
    #         deltas = deltas / (self.training_info["e_sigma"] ** 2)
    #     elif self.pixel_dist == "bernoulli":
    #         deltas = deltas / ((preds_old * (1 - preds_old)) + 1e-6)
    #
    #     h_mstep = self.rnem_cell.run_improvement(deltas, h_old, d_preds_old, gamma_old)
    #     mu_preds, depth_preds = self.rnem_cell.decode(h_mstep)
    #
    #     mu_preds = mu_preds.view([batch_size, -1] + list(mu_preds.size()[1:]))
    #     depth_preds = depth_preds.view([batch_size, -1] + list(depth_preds.size()[1:]))
    #     return h_mstep, mu_preds, depth_preds, torch.nn.functional.softmax(depth_preds, dim=1).detach()

    def improve(self, input_image, state, target_image=None):
        batch_size = input_image.size()[0]
        h_old, preds_old, d_preds_old, gamma_old = state
        deltas = input_image - preds_old
        if self.pixel_dist == "gaussian":
            deltas = deltas / (self.training_info["e_sigma"] ** 2)
        elif self.pixel_dist == "bernoulli":
            deltas = deltas / ((preds_old * (1 - preds_old)) + 1e-6)

        h_mstep = self.rnem_cell.run_improvement(deltas, h_old, d_preds_old, gamma_old) #Run encoders
        mu_preds, depth_preds = self.rnem_cell.decode(h_mstep) #Run decoders

        mu_preds = mu_preds.view([batch_size, -1] + list(mu_preds.size()[1:]))
        depth_preds = depth_preds.view([batch_size, -1] + list(depth_preds.size()[1:]))

        gamma = self.e_step(mu_preds, target_image, depth_preds) #Equal to depth_probs if target_data not given

        return h_mstep, mu_preds, depth_preds, gamma.detach()

    # def forward(self, inputs, state, actions, run_static_img):
    #     input_data, target_data = inputs  # RV: input_data=t'th noisy data, target_data=t+1'th true data
    #     # Both of shape (B, 1, W, H, C)
    #     h_old, preds_old, d_preds_old, gamma_old = state
    #
    #     h_new, mu_preds, depth_preds, gamma = self.improve(input_data, state, target_data)
    #     if not run_static_img:
    #         h_new, mu_preds, depth_preds, depth_probs = self.forward_rollout(h_new, actions, list(preds_old.size()))
    #         gamma = self.e_step(mu_preds, target_data, depth_preds)
    #
    #     return h_new, mu_preds, depth_preds, gamma.detach()

    def physics(self, target_shape, target_data, state, actions):
        #target_shape: (B, ?, W, H, C)
        h_old, preds_old, d_preds_old, gamma_old = state

        h_new, mu_preds, depth_preds = self.rnem_cell.forward_rollout(h_old, actions) #Run physics with decoders
        mu_preds = mu_preds.view([target_shape[0], -1] + target_shape[2:])
        depth_preds = depth_preds.view([target_shape[0], -1] + target_shape[2:-1] + [1])

        gamma = self.e_step(mu_preds, target_data, depth_preds) #Equal to depth_probs if target_data not given
        return h_new, mu_preds, depth_preds, gamma.detach()


    def do_random_rollout(self):
        return np.random.random_sample() < self.rollout_prob

    def do_random_static(self):
        return np.random.random_sample() < self.static_prob

    def do_improve_not_physics(self):
        return np.random.random_sample() < self.improve_prob


###################################End building the recurrent interaction network#############################



def compute_prior(distribution, pixel_prior):
    if distribution == 'bernoulli':
        return pixel_prior['p']
    elif distribution == 'gaussian':
        return pixel_prior['mu']
    else:
        raise KeyError('Unknown distribution: "{}"'.format(distribution))

# log bci
def binomial_cross_entropy_loss(y, t):
    clipped_y = torch.clamp(y, 1e-6, 1 - 1e-6)
    return -(t * torch.log(clipped_y) + (1.0 - t) * torch.log(1.0 - clipped_y))

# log gaussian
def gaussian_squared_error_loss(mu, sigma, x):
    clipped = torch.FloatTensor([np.clip(sigma ** 2, 1e-6, 1e6)]).to(ptu.device)
    return ((torch.pow(mu - x, 2)) / (2 * clipped)) + torch.FloatTensor([np.log(np.clip(sigma, 1e-6, 1e6))]).to(ptu.device)

# compute KL(p1, p2)
def kl_loss_bernoulli(p1, p2):
    tmp1 = p1 * torch.log(torch.clamp(p1/torch.clamp(p2, 1e-6, 1 - 1e-6), 1e-6, 1 - 1e-6))
    tmp2 = (1-p1) * torch.log(torch.clamp((1-p1)/torch.clamp(1-p2, 1e-6, 1 - 1e-6), 1e-6, 1 - 1e-6))
    return tmp1 + tmp2

# compute KL(p1, p2)
def kl_loss_gaussian(mu1, mu2, sigma1, sigma2):
    return torch.FloatTensor([np.log(np.clip(sigma2/sigma1, 1e-6, 1e6))]).to(ptu.device) \
           + (sigma1 ** 2 + torch.pow(mu1 - mu2, 2)) / (2 * sigma2 ** 2) - 0.5


def compute_outer_loss(pred, gamma, d_preds, target, prior, training_info):
    pixel_distribution = training_info["pixel_dist"]
    loss_inter_weight = training_info["loss_inter_weight"]
    include_kl = training_info["include_kl"]
    #target: (B, K, W, H, C)

    if pixel_distribution == 'bernoulli':
        intra_loss = binomial_cross_entropy_loss(pred, target)
        inter_loss = kl_loss_bernoulli(prior, pred)
    elif pixel_distribution == 'gaussian':
        # MC: this returns the negative log likelihood, which we want to minimize
        intra_loss = gaussian_squared_error_loss(pred, 1.0, target)
        inter_loss = kl_loss_gaussian(pred, prior, 1.0, 1.0)
    else:
        raise KeyError('Unknown pixel_distribution: "{}"'.format(pixel_distribution))


    # weigh losses by gamma and reduce by taking mean across B and sum across H, W, C, K
    # implemented as sum over all then divide by B
    batch_size = target.size()[0]
    # # compute normal losses
    # intra_loss = torch.sum(intra_loss * gamma.detach()) / batch_size
    # inter_loss = torch.sum(inter_loss * (1. - gamma.detach())) / batch_size

    # MC: these are scalars.
    intra_loss = torch.sum(intra_loss * gamma) / batch_size
    inter_loss = torch.sum(inter_loss * (1. - gamma)) / batch_size

    if training_info["include_depth"]:
        d_probs = torch.nn.functional.softmax(d_preds, dim=1)  # RV: (B, K, W, H, 1)
        log_d_probs = -torch.log(torch.clamp(d_probs, 1e-6, 1 - 1e-6))
        intra_d_prob_loss = torch.sum(log_d_probs * gamma) / batch_size
        inter_d_prob_loss = torch.sum(log_d_probs * (1. - gamma)) / batch_size
        # intra_loss = torch.sum((intra_loss + log_d_probs) * gamma) / batch_size
        # inter_loss = torch.sum((inter_loss + log_d_probs) * (1. - gamma)) / batch_size
    else:
        intra_d_prob_loss = torch.tensor(0, dtype=torch.float32).to(ptu.device)
        inter_d_prob_loss = torch.tensor(0, dtype=torch.float32).to(ptu.device)
    # else:
    #     # compute normal losses
    #     intra_loss = torch.sum(intra_loss * gamma) / batch_size
    #     inter_loss = torch.sum(inter_loss * (1. - gamma)) / batch_size

    intra_total_loss = intra_loss + intra_d_prob_loss
    inter_total_loss = inter_loss + inter_d_prob_loss
    total_loss = intra_total_loss  #+ loss_inter_weight * inter_loss
    if include_kl:
        total_loss += loss_inter_weight * inter_total_loss

    results = {
        'intra_pred_loss': intra_loss,
        'intra_d_prob_loss': intra_d_prob_loss,
        'intra_total_loss': intra_total_loss,
        'inter_pred_loss': inter_loss,
        'inter_d_pred_loss': inter_d_prob_loss,
        'inter_total_loss': inter_total_loss,
        'total_loss' : total_loss
    }
    # print(results)
    return results

def log_likelihood_loss(pred, d_preds, target, prior, training_info):
    pixel_distribution = training_info["pixel_dist"]
    loss_inter_weight = training_info["loss_inter_weight"]
    include_kl = training_info["include_kl"]

    if training_info["include_depth"]:
        d_probs = torch.nn.functional.softmax(d_preds, dim=1)  # RV: (B, K, W, H, 1)
        pred = torch.sum(pred * d_probs, dim=1, keepdim=True)

    if pixel_distribution == 'bernoulli':
        intra_ub_loss = binomial_cross_entropy_loss(pred, target)
        inter_ub_loss = kl_loss_bernoulli(prior, pred)
    elif pixel_distribution == 'gaussian':
        # intra_ub_loss = gaussian_squared_error_loss(pred, 1.0, target)
        intra_ub_loss = torch.nn.functional.mse_loss(pred, target, size_average=True)
        inter_ub_loss = kl_loss_gaussian(pred, prior, 1.0, 1.0)
    else:
        raise KeyError('Unknown pixel_distribution: "{}"'.format(pixel_distribution))

    # weigh losses by gamma and reduce by taking mean across B and sum across H, W, C, K
    # implemented as sum over all then divide by B
    batch_size = target.size()[0]

    # compute normal losses
    intra_ub_loss = torch.sum(intra_ub_loss) / batch_size
    inter_ub_loss = torch.sum(inter_ub_loss) / batch_size
    total_ub_loss = intra_ub_loss #+ loss_inter_weight * inter_ub_loss

    if include_kl:
        total_ub_loss += loss_inter_weight * inter_ub_loss

    return total_ub_loss, intra_ub_loss, inter_ub_loss, pred


def compute_depth_loss(d_pred, mu_pred, target, training_info):
    # if not training_info["compute_depth_loss"]:i
    #     return torch.tensor(0, dtype=torch.float32).to(ptu.device)
    batch_size = target.size()[0]
    # depth_probs = torch.nn.functional.softmax(d_pred, dim=1) #RV: (B, K, W, H, 1)
    depth_penalty = torch.sum(d_pred) / (batch_size * training_info['k'])
    return depth_penalty

def gamma_loss(d_pred, gamma):
    batch_size = d_pred.size()[0]
    depth_probs = torch.nn.functional.softmax(d_pred, dim=1)  # RV: (B, K, W, H, 1)
    loss = binomial_cross_entropy_loss(depth_probs, gamma) #Note, may need to change if k>2
    loss = torch.sum(loss) / batch_size
    return loss

def theta_loss(thetas, training_info):
    batch_size = thetas.size()[0] / training_info['k']
    loss = torch.sum(torch.pow(thetas, 2)) * training_info["theta_loss_weight"] / batch_size
    return loss


######################################Start running RNEM######################################
def new_nem_iterations(nem_cell, training_info, input_data, target_data, actions, k, device, groups=None):
    input_shape = input_data.size()
    # Note input shape should be 6D input (T, B, K, W, H, C)
    # Action: (T, B, C)
    assert len(input_shape) == 6
    W, H, C = input_shape[-3:]

    # input_layer_info, rnem_layer_info, output_layer_info = cfg()

    # compute prior
    # prior = 0.0 # RV: Prior is of shape (1, 1, 1, 1, 1) with value 0.0
    prior = compute_prior(distribution=training_info["pixel_dist"], pixel_prior=training_info["pixel_prior"])

    hidden_state = nem_cell.init_state(input_shape[1], k, dtype=torch.float32)
    # print(torch.max(hidden_state[2]), torch.min(hidden_state[2]), torch.mean(hidden_state[2]))
    # RV: hidden_state=(h=batch_size*k zero states, pred=all zeros, gamma=random assignments) Why?!?!
    # RV: Shouldn't hidden state just be h and not a tuple containing the other stuff
    # MC: we initialize the the theta to be 0, although it could be anything.
    # MC: we initialize pred to be 0s (assuming we have a black background), just as our prior
    # MC: we initialize the gamma to be random to help break symmetry between the cluster assignments.

    all_losses = {
        'intra_pred_loss': [],
        'intra_d_prob_loss': [],
        'intra_total_loss': [],
        'inter_pred_loss': [],
        'inter_d_pred_loss': [],
        'inter_total_loss': [],
        'total_loss': [],
        'log_likihood_intra_loss': [],
        'log_likihood_inter_loss': [],
        'log_likihood_total_loss': [],
        'depth_penalty': [],
        'gamma_loss': [],
        'theta_loss': []
    }


    # build static iterations
    outputs = [hidden_state]
    # total_losses, total_ub_losses, r_total_losses, r_total_ub_losses, other_losses, other_ub_losses, r_other_losses, r_other_ub_losses = [], [], [], [], [], [], [], []
    # depth_pred_losses, depth_penalities = [], []
    nr_steps, loss_step_weights = training_info["nr_steps"], training_info["loss_step_weights"]

    if loss_step_weights == "all":
        loss_step_weights = ptu.ones(nr_steps)
    elif loss_step_weights == "last":
        loss_step_weights = ptu.zeros(nr_steps)
        loss_step_weights[-1] = 1
    else:
        raise KeyError('Unknown loss_step_weight type: "{}"'.format(loss_step_weights))

    # if actions is None:
    #     actions = [None] * len(target_data)

    frame_at = 0
    run_static_img = nem_cell.do_random_static()
    if run_static_img:
        frame_at = np.random.randint(len(target_data))

    input_frame_history = []
    target_frame_history = []
    for step_at, loss_weight in enumerate(loss_step_weights):
        #Do a "rollout" (e.g. pretend we do not have the true image and gamma)
        if step_at >= 5 and nem_cell.do_random_rollout():
            d_probs = torch.nn.functional.softmax(hidden_state[2], dim=1)
            prev_img = torch.sum(hidden_state[1] * d_probs, dim=1, keepdim=True)
            hidden_state = hidden_state[0], hidden_state[1], hidden_state[2], d_probs.detach() #Gamma becomes the depth_probs
            input_frame_history.append(-1)
        else:
            prev_img = input_data[frame_at]
            input_frame_history.append(frame_at)

        #We do an improvement step if: 1. Static image  2. We are at the very beginning  3. Randomly choose to
        if run_static_img or step_at < training_info["initial_improve_steps"] or nem_cell.do_improve_not_physics():
            targ_img = target_data[frame_at]
            hidden_state = nem_cell.improve(prev_img, hidden_state, targ_img)
            target_frame_history.append(frame_at)
        else: #Do physics otherwise
            if frame_at+1 == len(target_data): #If we are at the last image, the target image is the same image
                targ_img = target_data[frame_at]
            else: #If not, predict the next image
                targ_img = target_data[frame_at+1]

            next_action = None
            if actions is not None and frame_at < actions.size()[0]:
                next_action = actions[frame_at]

            hidden_state = nem_cell.physics(list(targ_img.size()), targ_img, hidden_state, next_action)
            frame_at = min(frame_at+1, len(target_data)-1) #frame at +=1, and is at max len(target_data)-1
            target_frame_history.append(frame_at)

        h_new, mu_preds, d_preds, gamma = hidden_state

        # compute nem losses
        losses_dict = compute_outer_loss(mu_preds, gamma, d_preds, targ_img, prior,
                                         training_info)

        # compute estimated loss upper bound (which doesn't use E-step)
        total_ub_loss, intra_ub_loss, inter_ub_loss, _ = log_likelihood_loss(mu_preds, d_preds, targ_img, prior,
                                                                             training_info)
        losses_dict['log_likihood_intra_loss'] = intra_ub_loss
        losses_dict['log_likihood_inter_loss'] = inter_ub_loss
        losses_dict['log_likihood_total_loss'] = total_ub_loss

        depth_penalty = compute_depth_loss(d_preds, mu_preds, targ_img, training_info)
        losses_dict['depth_penalty'] = depth_penalty
        if training_info["extra_depth_penalty"]:
            losses_dict['total_loss'] += depth_penalty  # + depth_penalty

        g_loss = gamma_loss(d_preds, gamma)
        losses_dict['gamma_loss'] = g_loss
        if training_info['gamma_loss']:
            losses_dict['total_loss'] += g_loss

        #Loss on thetas
        hidden_loss = theta_loss(h_new, training_info)
        losses_dict['theta_loss'] = hidden_loss
        losses_dict['total_loss'] += hidden_loss

        for aKey in all_losses:
            all_losses[aKey].append(losses_dict[aKey] * loss_weight)

        hidden_state = h_new, mu_preds.detach(), d_preds.detach(), gamma.detach()
        outputs.append(hidden_state)


    thetas, preds, d_preds, gammas = zip(*outputs)
    thetas = torch.stack(thetas)
    preds = torch.stack(preds)
    d_preds = torch.stack(d_preds) # (31, 1, 2, 64, 64, 1)
    gammas = torch.stack(gammas)

    loss_weight_norm = torch.sum(loss_step_weights)
    # MC: sums across all timesteps
    for aKey in all_losses:
        all_losses[aKey] = torch.sum(torch.stack(all_losses[aKey])) / loss_weight_norm


    all_losses["thetas"] = thetas
    all_losses["preds"] = preds
    all_losses["gammas"] = gammas
    all_losses["d_preds"] = d_preds
    all_losses["input_frame_history"] = input_frame_history
    all_losses["target_frame_history"] = target_frame_history
    return all_losses


######################################Start running rollout######################################
def new_nem_rollouts(nem_cell, training_info, input_data, target_data, actions, k, device, groups=None):
    input_shape = input_data.size()
    # Note input shape should be 6D input (T, B, K, W, H, C)
    #Action: (T, B, A)
    assert len(input_shape) == 6

    # compute prior
    # prior = 0.0 # RV: Prior is of shape (1, 1, 1, 1, 1) with value 0.0
    prior = compute_prior(distribution=training_info["pixel_dist"], pixel_prior=training_info["pixel_prior"])

    hidden_state = nem_cell.init_state(input_shape[1], k, dtype=torch.float32)

    all_losses = {
        'true_loss': []
    }

    # build static iterations
    outputs = [hidden_state]

    rollout_length = min(training_info["nr_steps"], len(target_data)-1)
    # nr_steps = training_info["nr_steps"]

    TARGET_SIZE = list(input_data[0].size())
    input_frame_history = []
    target_frame_history = []

    #Initial seed steps on first image
    for i in range(training_info["initial_improve_steps"]):
        #Run hidden cell on single image
        hidden_state = nem_cell.improve(input_data[0], hidden_state, target_data[0])
        h_new, mu_preds, d_preds, gamma = hidden_state
        hidden_state = h_new, mu_preds.detach(), d_preds.detach(), gamma.detach()

        outputs.append(hidden_state)
        input_frame_history.append(0)
        target_frame_history.append(0)


    #Initial seed steps with physics on first roll_start images (lookahead here)
    for t in range(training_info["rollout_seed_steps"]):
        # Run physics
        next_action = None
        if actions is not None and t < len(actions):
            next_action = actions[t]

        hidden_state = nem_cell.physics(TARGET_SIZE, target_data[t + 1], hidden_state, next_action)
        h_new, mu_preds, d_preds, gamma = hidden_state

        #Run improvement steps certain number of times
        for i in range(training_info["rollout_out_improve_steps"]):
            hidden_state = nem_cell.improve(input_data[t+1], hidden_state, target_data[t+1])

        hidden_state = h_new, mu_preds.detach(), d_preds.detach(), gamma.detach()
        outputs.append(hidden_state)
        input_frame_history.append(t)
        target_frame_history.append(t+1)

    ########Rollout#######
    for t in range(training_info["rollout_seed_steps"], rollout_length):
        #Predict the next image just via physics
        next_action = None
        if actions is not None and t < len(actions):
            next_action = actions[t]

        hidden_state = nem_cell.physics(TARGET_SIZE, None, hidden_state, next_action)
        # for i in range(training_info["rollout_out_improve_steps"]):
        #     #Note that hidden_state[3] = gamma = depth_probs when no target image given!
        #     net_pred = torch.sum(hidden_state[1] * hidden_state[3], dim=1, keepdim=True)
        #     hidden_state = nem_cell.improve(net_pred, hidden_state, None)

        h_new, mu_preds, d_preds, gamma = hidden_state
        total_ub_loss, intra_ub_loss, inter_ub_loss, net_pred = log_likelihood_loss(mu_preds, d_preds,
                                                                                    input_data[t], prior, training_info)
        outputs.append(hidden_state)
        input_frame_history.append(t)
        target_frame_history.append(t + 1)

        all_losses['true_loss'].append(total_ub_loss)

    thetas, preds, d_preds, gammas = zip(*outputs)
    thetas = torch.stack(thetas)
    preds = torch.stack(preds)
    d_preds = torch.stack(d_preds)  # (31, 1, 2, 64, 64, 1)
    gammas = torch.stack(gammas)

    # MC: sums across al timesteps
    for aKey in all_losses:
        torch.stack(all_losses[aKey])
        all_losses[aKey] = torch.sum(torch.stack(all_losses[aKey])) / (rollout_length - training_info["rollout_seed_steps"])

    all_losses["thetas"] = thetas
    all_losses["preds"] = preds
    all_losses["gammas"] = gammas
    all_losses["d_preds"] = d_preds
    all_losses["input_frame_history"] = input_frame_history
    all_losses["target_frame_history"] = target_frame_history
    return all_losses

#####Stage 2 code#####
def generate_random_action(batch_size):
    #Note: true_action is a torch tensor of size 1, 13
    tmp = ptu.zeros((1, batch_size, 13))
    # tmp = torch.rand(tmp.size(), device = ptu.device) #Uniformly random between 0 and 1
    which_object = np.random.randint(0, 3, batch_size)
    tmp[0, :, :3] = 0
    tmp[0, :, which_object] = 1

    tmp[0, :, 4] = 0
    tmp[0, :, 7] = 0
    tmp[0, :, 8] = 0

    # tmp = sample_actions(n_actions) #B, A
    # tmp = np.expand_dims(tmp, 0) #1, B, A
    # return ptu.from_numpy(tmp)
    return tmp

def sample_actions(n_actions):
    # ## ply_ind, pos, axangle, scale, rgb
    # actions = [[0, [-.75, 0, 0], [0, 0, 1, 0], .4, [.75, .75, 0]],
    #            [0, [-.75, 0, 1], [0, 0, 1, 0], .4, [.25, .75, .25]],
    #            [0, [.75, 0, 0], [0, 0, 1, 0], .4, [.5, .25, 1]],
    #            [0, [.75, 0, 1], [0, 0, 1, 0], .4, [1, .25, .5]],
    #            [1, [0, 0, 2], [0, 0, 1, 0], .4, [.85, .25, 0]],
    #            [2, [0, 0, 3], [0, 0, 1, math.pi / 4], .4, [0, .75, .75]],

    ply_idx = np.random.randint(0, 3, size=(n_actions, 1))
    pos = np.random.uniform(low=[-1, 0, 0], high=[1, 0, 3], size=(n_actions, 3))
    axangle = np.random.uniform(low=[0, 0, 1, 0], high=[0, 0, 1, 0], size=(n_actions, 4))
    scale = np.random.uniform(low=0.4, high=0.4, size=(n_actions, 1))
    # rgb = np.random.uniform(low=[0, .25, 0], high=[1, .75, 1], size=(n_actions, 3))
    colors = [[.75, .75, 0],
              [.25, .75, .25],
              [.5, .25, 1],
              [1, .25, .5],
              [.85, .25, 0],
              [0, .75, .75]]
    rgb = np.random.randint(0, len(colors), size=(n_actions))
    rgb = np.array([colors[x] for x in rgb])
    actions = np.concatenate([ply_idx, pos, axangle, scale, rgb], -1)
    actions = [x for x in actions]
    return actions

def rollout_score(target_thetas, rollout_thetas, k):
    #Thetas:
    # print("thetas size: ", target_thetas.size(), rollout_thetas.size())
    #torch.Size([k, H]) torch.Size([k, H])
    total_loss = []
    for targ_ind in range(k):
        # loss = torch.sum(torch.pow(thetas, 2)) * training_info["theta_loss_weight"] / batch_size
        losses = torch.sum(torch.pow(rollout_thetas - target_thetas[targ_ind], 2), 1) #Broadcastings
        #losses
        total_loss.append(torch.min(losses))
    return torch.sum(torch.stack(total_loss))

def raw_results_to_list(results, k):
    d_probs = torch.nn.functional.softmax(results["d_preds"][-1], dim=1)  # RV: (1, K, W, H, 1)
    final_image = torch.sum(results["preds"][-1] * d_probs, dim=1, keepdim=True)  # RV: (1, W, H, 3)
    # print(d_probs.size(), final_image.size(), results["preds"].size())
    #torch.Size([1, 2, 64, 64, 1]) torch.Size([1, 1, 64, 64, 3]) torch.Size([11, 1, 2, 64, 64, 3])

    list_of_plots = [final_image]
    for ind in range(k):
        list_of_plots.append(results["preds"][-1, 0, ind])  # Adding mu predictions
    for ind in range(k):
        list_of_plots.append(results["d_preds"][-1, 0, ind])  # Adding depth predictions
    for ind in range(k):
        list_of_plots.append(d_probs[-1, ind])  # Adding depth probabilities
    return list_of_plots


def mpc_stuff(file_prefix, nem_cell, training_info, images, actions, k, device, groups=None):
    #Input data should have T = 1, same with target_data (ie. both should be single images)
    #(T, B, 1, W, H, 1)
    #Actions size: (T, B, A)
    batch_size = images.size()[1]
    num_actions_to_try = 6
    initial_image = images[:1]
    target_image = images[-1:]

    prev_static_prob = nem_cell.static_prob
    nem_cell.static_prob = 1
    results = new_nem_iterations(nem_cell, training_info, target_image, target_image, actions, k, device, groups)
    nem_cell.static_prob = prev_static_prob
    target_thetas = results["thetas"][-1] #(NR_steps, B, H) -> (B, H)

    if file_prefix is not None:
        list_of_plots = [target_image]
        list_of_plots.extend(raw_results_to_list(results, k))
        cur_fig, axes = plt.subplots(nrows=3*k+2, ncols=num_actions_to_try, figsize=(2 * (3*k+2), 2 * num_actions_to_try))
        cur_fig, axes = ptu.plot_columns(list_of_plots, 0, num_actions_to_try+1, (cur_fig, axes))
    del results

    num_actions_to_try -= 1
    all_scores = ptu.zeros(num_actions_to_try)
    final_images = []
    all_actions = []
    # all_actions = ptu.zeros([num_actions_to_try] + list(actions.size()))
    for i in range(num_actions_to_try):
        # rand_actions = generate_random_action(actions)
        rand_actions = generate_random_action(batch_size) #Action: (T=1, B, A)
        if i == 0:
            rand_actions = actions
        all_actions.append(rand_actions)
        #nem_cell, training_info, input_data, target_data, actions, k, device, groups=None
        results = new_nem_rollouts(nem_cell, training_info, images, images, rand_actions, k, device, groups)
        all_scores[i] = rollout_score(target_thetas, results["thetas"][-1], k)

        if file_prefix is not None:
            d_probs = torch.nn.functional.softmax(results["d_preds"], dim=1)  # RV: (1, K, W, H, 1)
            final_images.append(torch.sum(results["preds"] * d_probs, dim=1, keepdim=True))

            list_of_plots = [initial_image]
            list_of_plots.extend(raw_results_to_list(results, k))

            cur_fig, axes = ptu.plot_columns(list_of_plots, i+1, num_actions_to_try + 1, (cur_fig, axes))
            axes[0, i + 1].set_title("{:.0f}".format(all_scores[i].detach().cpu().numpy()))

    if file_prefix is not None:
        log_dir = training_info["log_path"]
        # cur_fig.suptitle('Sample {},  ARI Score: {:.3f}'.format(an_image_index, results["ari_score"]))
        cur_fig.savefig(os.path.join(log_dir, 'actions_{}.png'.format(file_prefix)),
                    bbox_inches='tight', pad_inches=0)
        plt.close(cur_fig)

    all_results = {
        "scores": all_scores[0]
    }
    return all_results


def mpc_generate_images(file_prefix, our_data_loader, nem_cell, training_info, k, sample_indices, device):
    for an_image_index in sample_indices:
        the_sample = our_data_loader.dataset[an_image_index]

        images = torch.from_numpy(the_sample["features"])  # (T, 1, W, H, 1)
        images = torch.unsqueeze(images, 1)  # (T, B=1, 1, W, H, 1)
        images = images.to(device)

        groups = torch.from_numpy(the_sample["groups"]) # (T, 1, W, H, 1)
        groups = torch.unsqueeze(groups, 1)  # (T, B=1, 1, W, H, 1)

        actions = None
        if our_data_loader.dataset.has_actions:
            actions = torch.from_numpy(the_sample["actions"])  # (T, A)
            actions = torch.unsqueeze(actions, 1)  # (T, B=1, C)
            actions = actions.view([actions.size()[0], actions.size()[1], actions.size()[3]])  # (T, B, 1, A) -> (T, B, A)
            actions = actions.to(device)

        results = mpc_stuff(file_prefix, an_image_index, nem_cell, training_info, images, actions, k, device, groups=groups)



