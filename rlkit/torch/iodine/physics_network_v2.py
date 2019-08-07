import torch
import torch.utils.data
from rlkit.torch.pytorch_util import from_numpy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch.networks import Mlp
from rlkit.torch import pytorch_util as ptu
import numpy as np
import pdb

###Maps name to a tuple (class type, lambda function defining model architecture)
Physics_Args = dict(
    reg_ac32 = ("reg",
        lambda det_size, sto_size, action_size: dict(
        det_size = det_size,
        sto_size = sto_size,
        action_size = action_size,
        action_enc_size = 32,
        hidden_activation=nn.ELU())
    ),
    mlp_ac32 = ("mlp",
        lambda repsize, action_size: dict(
        representation_size=repsize,
        action_size=action_size,
        action_enc_size=32,
        hidden_activation=nn.ELU())
    )
)


class PhysicsNetwork_v2(nn.Module):
    def __init__(
            self,
            det_size,
            sto_size,
            action_size,
            action_enc_size,
            hidden_activation=nn.ELU(),
            deterministic_state_activation=nn.ELU(),
            lambda_output_activation=identity
    ):
        super().__init__()
        self.K = None
        self.det_size = det_size
        self.sto_size = sto_size
        self.full_rep_size = det_size + sto_size
        self.action_size = action_size

        self.action_enc_size = action_enc_size if action_size > 0 else 0
        self.effect_size = action_enc_size
        hidden_size = self.full_rep_size
        self.interaction_size = self.full_rep_size

        self.inertia_encoder = Mlp((hidden_size,), self.full_rep_size, self.full_rep_size,
                                  hidden_activation=hidden_activation, output_activation=hidden_activation)
        #Note: Input to inertia_encoder is (B,Rd+Rs) as we take in the full state and not just the stochastic part

        #Action networks
        if action_size > 0:
            self.action_encoder = Mlp((hidden_size,), self.action_enc_size, action_size,
                                      hidden_activation=hidden_activation, output_activation=hidden_activation)
            self.action_effect_network = Mlp((hidden_size,), self.full_rep_size, self.action_enc_size + self.full_rep_size,
                                             hidden_activation=hidden_activation, output_activation=hidden_activation)
            self.action_attention_network = Mlp((hidden_size,), 1, self.action_enc_size + self.full_rep_size,
                                                hidden_activation=hidden_activation, output_activation=nn.Sigmoid())

        self.pairwise_encoder_network = Mlp((hidden_size*2,), self.interaction_size, self.full_rep_size*2,
                                     hidden_activation=hidden_activation, output_activation=hidden_activation)
        self.interaction_effect_network = Mlp((hidden_size,), self.effect_size, self.interaction_size,
                                              hidden_activation=hidden_activation, output_activation=hidden_activation)
        self.interaction_attention_network = Mlp((hidden_size,), 1, self.interaction_size,
                                                 hidden_activation=hidden_activation, output_activation=nn.Sigmoid())


        ###If deterministic state directly produces lambdas
        # if self.det_size != 0:
        #     # Note: final_merge_network consolidates all the information into the new deterministic state
        #     self.final_merge_network = Mlp((hidden_size,), self.det_size, self.effect_size+self.full_rep_size,
        #                                    hidden_activation=hidden_activation, output_activation=deterministic_state_activation)
        #     output_size = self.det_size
        # else:
        #     self.final_merge_network = Mlp((hidden_size,), self.sto_size, self.effect_size + self.full_rep_size,
        #                                    hidden_activation=hidden_activation, output_activation=hidden_activation)
        #     output_size = self.sto_size
        # self.state_to_lambdas1 = Mlp((hidden_size,), self.sto_size, output_size, hidden_activation=hidden_activation,
        #                             output_activation=lambda_output_activation)
        # self.state_to_lambdas2 = Mlp((hidden_size,), self.sto_size, output_size, hidden_activation=hidden_activation,
        #                             output_activation=lambda_output_activation)

        ###If deterministic state has a separate output branch
        self.final_merge_network = Mlp((hidden_size,), self.full_rep_size, self.effect_size + self.full_rep_size,
                                       hidden_activation=hidden_activation, output_activation=hidden_activation)
        if self.det_size != 0:
            self.det_output = Mlp((hidden_size,), self.det_size, self.full_rep_size,
                                  hidden_activation=hidden_activation, output_activation=deterministic_state_activation)
        self.lambdas1_output = Mlp((hidden_size,), self.sto_size, self.full_rep_size,
                                  hidden_activation=hidden_activation, output_activation=lambda_output_activation)
        self.lambdas2_output = Mlp((hidden_size,), self.sto_size, self.full_rep_size,
                                  hidden_activation=hidden_activation, output_activation=lambda_output_activation)

    def set_k(self, k):
        self.K = k

    #Inputs: sampled_state (B*K, Rd+Rs), Action: (B*K, A)
    # Note: sampled_state is Rd+Rs as we take in the full state and not just the stochastic part (Rs)
    def forward(self, sampled_state, actions):
        K = self.K
        bs = sampled_state.shape[0]//K

        state_enc_flat = self.inertia_encoder(sampled_state) #Encode sample

        if actions is not None:
            if self.action_size == 4 and actions.shape[-1] == 6:
                action_enc = self.action_encoder(actions[:, torch.LongTensor([0, 1, 3, 4])]) #RV: Encode actions, why torch.longTensor?
            else:
                action_enc = self.action_encoder(actions) #Encode actions
            state_enc_actions = torch.cat([state_enc_flat, action_enc], -1)

            state_action_effect = self.action_effect_network(state_enc_actions) #(bs*k, h)
            state_action_attention = self.action_attention_network(state_enc_actions) #(bs*k, 1)
            state_enc = (state_action_effect*state_action_attention).view(bs, K, self.full_rep_size) #(bs, k, h)
        else:
            state_enc = state_enc_flat.view(bs, K, self.full_rep_size) #(bs, k, h)

        if K != 1:
            pairs = []
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    pairs.append(torch.cat([state_enc[:, i], state_enc[:, j]], -1))  #Create array of all pairs

            all_pairs = torch.stack(pairs, 1).view(bs*K,  K-1, -1) #Create torch of all pairs

            pairwise_interaction = self.pairwise_encoder_network(all_pairs) #(bs*k,k-1,h)
            effect = self.interaction_effect_network(pairwise_interaction)  # (bs*k,k-1,h)
            attention = self.interaction_attention_network(pairwise_interaction)  #(bs*k,k-1,1)
            total_effect = (effect*attention).sum(1)  #(bs*k,h)
        else:
            total_effect = ptu.zeros((bs, self.effect_size)).to(sampled_state.device)

        state_and_effect = torch.cat([state_enc.view(bs*K, self.full_rep_size), total_effect], -1)  # (bs*k,h)

        ###If deterministic state directly produces lambdas
        # deter_state = self.final_merge_network(state_and_effect) #Deterministic state (bs*k,h)
        # lambdas1 = self.state_to_lambdas1(deter_state) #Initial lambda parameters (B*K,R)
        # lambdas2 = self.state_to_lambdas2(deter_state) #(B*K,R)
        # if self.det_size == 0:
        #     deter_state = None

        aggregate_state = self.final_merge_network(state_and_effect)
        if self.det_size == 0:
            deter_state = None
        else:
            deter_state = self.det_output(aggregate_state)
        lambdas1 = self.lambdas1_output(aggregate_state)
        lambdas2 = self.lambdas2_output(aggregate_state)

        return deter_state, lambdas1, lambdas2



class PhysicsNetworkMLP_v2(nn.Module):
    def __init__(
            self,
            K,
            representation_size,
            action_size,
            action_enc_size,
    ):
        super().__init__()
        self.K = K
        self.rep_size = representation_size
        self.action_size = action_size

        self.action_enc_size = action_enc_size if action_size > 0 else 0
        self.effect_size = action_enc_size
        # self.enc_rep_size = representation_size - self.effect_size
        hidden_size = representation_size
        self.interaction_size = representation_size

        self.action_encoder = Mlp((hidden_size,), self.action_enc_size, action_size,
                                  hidden_activation=nn.ELU(), output_activation=nn.ELU())
        self.mlp_net = Mlp([hidden_size]*5, self.rep_size*K, self.rep_size*K+self.action_enc_size,
                                             hidden_activation=nn.ELU())


    # lambdas are each (bs*K, representation_size), actions are (bs*K, A)
    def forward(self, lambda1, lambda2, actions):
        K = self.K
        # lambda1 = lambda1.view(-1, K, self.rep_size) #(bs, K, rep_size)
        lambda1 = lambda1.view(-1, K*self.rep_size) #(bs, K*rep_size)

        # pdb.set_trace()
        actions = actions[::K]
        if self.action_size == 4:
            action_enc = self.action_encoder(actions[:, torch.LongTensor([0, 1, 3, 4])])  # RV: Encode actions, why torch.longTensor?
        else:
            action_enc = self.action_encoder(actions)  # Encode actions
        lambda1 = torch.cat([lambda1, action_enc], -1)  # RV: Concatonate lambdas with actions
        lambda1 = self.mlp_net(lambda1)
        # lambda1 = lambda1.view(-1, K, self.rep_size)
        lambda1 = lambda1.view(-1, self.rep_size)

        return lambda1, lambda2

    def set_k(self, k):
        assert k == self.K  #We cannot change k as mlp takes in a fixed number of latents
