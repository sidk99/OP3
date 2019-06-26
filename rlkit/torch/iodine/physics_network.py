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

class PhysicsNetwork(nn.Module):
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

        #New version
        self.lambda_encoder = Mlp((hidden_size,), self.rep_size, self.rep_size,
                                  hidden_activation=nn.ELU(), output_activation=nn.ELU())

        #Action networks
        if action_size > 0:
            self.action_encoder = Mlp((hidden_size,), self.action_enc_size, action_size,
                                      hidden_activation=nn.ELU(), output_activation=nn.ELU())
            self.action_effect_network = Mlp((hidden_size,), self.rep_size, self.action_enc_size + self.rep_size,
                                             hidden_activation=nn.ELU(), output_activation=nn.ELU())
            self.action_attention_network = Mlp((hidden_size,), 1, self.action_enc_size + self.rep_size,
                                                hidden_activation=nn.ELU(), output_activation=nn.Sigmoid())

        self.pairwise_encoder_network = Mlp((hidden_size*2,), self.interaction_size, self.rep_size*2,
                                     hidden_activation=nn.ELU(), output_activation=nn.ELU())
        self.interaction_effect_network = Mlp((hidden_size,), self.effect_size, self.interaction_size,
                                              hidden_activation=nn.ELU(), output_activation=nn.ELU())
        self.interaction_attention_network = Mlp((hidden_size,), 1, self.interaction_size,
                                                 hidden_activation=nn.ELU(), output_activation=nn.Sigmoid())

        self.final_merge_network = Mlp((hidden_size,), self.rep_size, self.effect_size+self.rep_size, hidden_activation=nn.ELU())

        # #Old version
        # self.lambda_encoder = Mlp((128,), self.rep_size, self.rep_size,
        #                           hidden_activation=nn.ELU())
        #
        # #Action networks
        # if action_size > 0:
        #     self.action_encoder = Mlp((128,), self.action_enc_size, action_size,
        #                               hidden_activation=nn.ELU())
        #     self.action_attention_network = Mlp((128,), self.rep_size, self.action_enc_size + self.rep_size,
        #                                         hidden_activation=nn.ELU(), output_activation=nn.Sigmoid())
        #     self.action_effect_network = Mlp((128,), 1, self.action_enc_size + self.rep_size,
        #                                      hidden_activation=nn.ELU())
        #
        # self.pairwise_encoder_network = Mlp((256,), self.interaction_size, self.rep_size*2,
        #                              hidden_activation=nn.ELU(), output_activation=nn.ELU())
        # self.interaction_effect_network = Mlp((128,), self.effect_size, self.interaction_size,
        #                                       hidden_activation=nn.ELU(), output_activation=nn.ELU())
        # self.interaction_attention_network = Mlp((128,), 1, self.interaction_size,
        #                                          hidden_activation=nn.ELU(), output_activation=nn.Sigmoid())
        #
        # self.final_merge_network = Mlp((128,), self.rep_size, self.effect_size+self.rep_size, hidden_activation=nn.ELU())


    def forward(self, lambda1, lambdas2, actions):
        # input is (bs*K, representation_size)
        K = self.K

        lambda1_enc_flat = self.lambda_encoder(lambda1) #Encode initial lambdas

        if actions is not None:
            if self.action_size == 4:
                action_enc = self.action_encoder(actions[:, torch.LongTensor([0, 1, 3, 4])]) #RV: Encode actions, why torch.longTensor?
            else:
                action_enc = self.action_encoder(actions) #Encode actions
            lambda1_enc_actions = torch.cat([lambda1_enc_flat, action_enc], -1) #RV: Concatonate lambdas with actions?
            # lambda1_enc_actions = lambda1_enc_actions.view(-1, K, self.rep_size + self.action_enc_size)
            # lambda1_enc = lambda1_enc_actions.view(-1, K, self.enc_rep_size + self.action_enc_size) #bs, k, h

            lambda1__action_effect = self.action_effect_network(lambda1_enc_actions) #(bs*k, h)
            lambda1__action_attention = self.action_attention_network(lambda1_enc_actions) #(bs*k, 1)
            lambda1_enc = (lambda1__action_effect*lambda1__action_attention).view(-1, K, self.rep_size) #(bs, k, h)
        else:
            lambda1_enc = lambda1_enc_flat.view(-1, K, self.rep_size)
            # lambda1_enc = lambda1_enc_flat.view(-1, K, self.enc_rep_size)  #bs, k, h

        bs = lambda1_enc.shape[0]

        if K != 1:
            pairs = []
            for i in range(K):
                for j in range(K):
                    if i == j:
                        continue
                    pairs.append(torch.cat([lambda1_enc[:, i], lambda1_enc[:, j]], -1))  #Create array of all pairs

            all_pairs = torch.stack(pairs, 1).view(bs*K,  K-1, -1) #Create torch of all pairs

            pairwise_interaction = self.pairwise_encoder_network(all_pairs) #(bs*k,k-1,h)
            effect = self.interaction_effect_network(pairwise_interaction)  # (bs*k,k-1,h)
            attention = self.interaction_attention_network(pairwise_interaction)  #(bs*k,k-1,1)
            total_effect = (effect*attention).sum(1)  #(bs*k,h)
        else:
            total_effect = ptu.zeros((bs, self.effect_size))

        lambda_and_effect = torch.cat([lambda1_enc_flat, total_effect], -1)  # (bs*k,h)
        new_lambdas = self.final_merge_network(lambda_and_effect) #(bs*k,h)

        #Old version
        # interaction = self.embedding_network(all_pairs)  #(bs*k,k-1,h), input size is (h)?
        # effect = self.effect_network(interaction) #(bs*k,k-1,h)
        # attention = self.attention_network(interaction) #(bs*k,k-1,1)
        # total_effect = (attention * effect).view(bs*K, (K-1), -1).sum(1) # TODO check this is right
        #
        # lambda_physics = self.encoder_network(total_effect)
        #
        # # lambdas2 = self.encoder2_network(lambdas2)
        # new_lambdas = torch.cat([lambda1_enc_flat, lambda_physics], -1)

        return new_lambdas, None

    def initialize_hidden(self, bs):
        return (Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))),
                Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))))


class PhysicsNetworkMLP(nn.Module):
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

        return lambda1, None


# class PhysicsNetwork(nn.Module):
#     def __init__(
#             self,
#             K,
#             representation_size,
#             action_size,
#             action_enc_size,
#     ):
#         super().__init__()
#         self.K = K
#         self.rep_size = representation_size
#         self.action_size = action_size
#
#         self.action_enc_size = action_enc_size if action_size > 0 else 0
#         self.effect_size = action_enc_size
#         self.enc_rep_size = representation_size - self.effect_size
#         self.interaction_size = 128
#
#         #self.action_encoder = Mlp((128,), self.action_enc_size, action_size, hidden_activation=nn.ELU())
#
#         if action_size > 0:
#             self.action_encoder = Mlp((128,), self.action_enc_size, action_size,
#                                       hidden_activation=nn.ELU())
#         self.lambda_encoder = Mlp((128, ), self.enc_rep_size, representation_size, hidden_activation=nn.ELU())
#
#         self.embedding_network = Mlp((256,), self.interaction_size, (self.enc_rep_size + self.action_enc_size)*2,
#                                      hidden_activation=nn.ELU(),
#                                      output_activation=nn.ELU())
#         self.effect_network = Mlp((128,), self.interaction_size, self.interaction_size, hidden_activation=nn.ELU(),
#                                      output_activation=nn.ELU())
#         self.attention_network = Mlp((128,), 1, self.interaction_size, hidden_activation=nn.ELU(),
#                                      output_activation=nn.Sigmoid())
#         self.encoder_network = Mlp((128,), self.effect_size, self.interaction_size, hidden_activation=nn.ELU())
#         #self.encoder2_network = Mlp((128,), representation_size, representation_size,
#         #                            hidden_activation=nn.ELU())
#
#
#
#     def forward(self, lambda1, lambdas2, actions):
#         # input is (bs*K, representation_size)
#         K = self.K
#
#         lambda1_enc_flat = self.lambda_encoder(lambda1)
#
#         if actions is not None:
#             if self.action_size == 4:
#                 action_enc = self.action_encoder(actions[:, torch.LongTensor([0, 1, 3, 4])])
#             else:
#                 action_enc = self.action_encoder(actions)
#             lambda1_enc_actions = torch.cat([lambda1_enc_flat, action_enc], -1)
#             lambda1_enc = lambda1_enc_actions.view(-1, K, self.enc_rep_size + self.action_enc_size)
#         else:
#             lambda1_enc = lambda1_enc_flat.view(-1, K, self.enc_rep_size)
#
#         bs = lambda1_enc.shape[0]
#
#         pairs = []
#         for i in range(K):
#             for j in range(K):
#                 if i == j:
#                     continue
#                 pairs.append(torch.cat([lambda1_enc[:, i], lambda1_enc[:, j]], -1))
#
#         all_pairs = torch.stack(pairs, 1).view(bs*K,  K-1, -1)
#
#         interaction = self.embedding_network(all_pairs)
#         effect = self.effect_network(interaction)
#
#         attention = self.attention_network(interaction)
#
#         total_effect = (attention * effect).view(bs*K, (K-1), -1).sum(1) # TODO check this is right
#
#         lambda_physics = self.encoder_network(total_effect)
#
#         #lambdas2 = self.encoder2_network(lambdas2)
#         new_lambdas = torch.cat([lambda1_enc_flat, lambda_physics], -1)
#
#         return new_lambdas, None
#
#     def initialize_hidden(self, bs):
#         return (Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))),
#                 Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))))