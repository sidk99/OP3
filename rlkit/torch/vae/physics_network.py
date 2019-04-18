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
from rlkit.torch.core import PyTorchModule

class PhysicsNetwork(PyTorchModule):
    def __init__(
            self,
            K,
            representation_size,
            action_size,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.K = K
        self.rep_size = representation_size
        self.action_size = action_size

        self.enc_size = enc_size = 128

        self.encoder_network = Mlp((256,), enc_size, representation_size, hidden_activation=nn.ELU())

        self.action_encoder = Mlp((128, ), enc_size, action_size, hidden_activation=nn.ELU())

        self.a_attention= Mlp((128,), 1, enc_size*2, hidden_activation=nn.Tanh(),
                                     output_activation=nn.Sigmoid())
        self.a_effect = Mlp((256,), enc_size, enc_size*2, hidden_activation=nn.ELU(),
                                  output_activation=nn.ELU())


        self.lns = nn.ModuleList([nn.LayerNorm(enc_size),
                                 nn.LayerNorm(enc_size),
                                  nn.LayerNorm(enc_size)]
                                 )

        self.embedding_network = Mlp((256,), enc_size, enc_size*2, hidden_activation=nn.ELU(),
                                     output_activation=nn.ELU())
        self.effect_network = Mlp((256,), enc_size, enc_size, hidden_activation=nn.ELU(),
                                     output_activation=nn.ELU())
        self.attention_network = Mlp((128,), 1, enc_size, hidden_activation=nn.Tanh(),
                                     output_activation=nn.Sigmoid())

        self.final_encoder = Mlp((256,), enc_size, enc_size*2, hidden_activation=nn.ELU())






    def forward(self, input, actions):
        # input is (bs*K, representation_size)
        K = self.K
        rep_size = self.rep_size
        enc_size = self.enc_size

        lambda_enc = self.encoder_network(input)
        action_enc = self.lns[0](self.action_encoder(actions))
        lambdas = torch.cat([lambda_enc, action_enc], -1)

        action_attention = self.a_attention(lambdas)
        action_effect = self.a_effect(lambdas)

        action_lambdas = action_attention * action_effect


        lambdas = action_lambdas.view(-1, K, enc_size)
        bs = lambdas.shape[0]

        pairs = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                pairs.append(torch.cat([lambdas[:, i], lambdas[:, j]], -1))

        all_pairs = torch.stack(pairs, 1).view(bs*K,  K-1, -1)

        interaction = self.lns[1](self.embedding_network(all_pairs))
        effect = self.effect_network(interaction)

        attention = self.attention_network(interaction)

        total_effect = (attention * effect).view(bs*K, (K-1), -1).sum(1) # TODO check this is right

        new_lambdas = torch.cat([action_lambdas, total_effect], -1) # TODO apply another network after this

        new_lambdas = self.final_encoder(new_lambdas)

        return new_lambdas

    def initialize_hidden(self, bs):
        return (Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))),
                Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))))