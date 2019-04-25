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

        enc_size = 128

        self.action_enc_size = 32

        #self.action_encoder = Mlp((128,), self.action_enc_size, action_size, hidden_activation=nn.ELU())

        self.embedding_network = Mlp((256,), enc_size, representation_size*2,
                                     hidden_activation=nn.ELU(),
                                     output_activation=nn.ELU())
        self.effect_network = Mlp((128,), enc_size, enc_size, hidden_activation=nn.ELU(),
                                     output_activation=nn.ELU())
        self.attention_network = Mlp((128,), 1, enc_size, hidden_activation=nn.ELU(),
                                     output_activation=nn.Sigmoid())
        self.encoder_network = Mlp((128,), representation_size, enc_size, hidden_activation=nn.ELU())
        self.encoder2_network = Mlp((128,), representation_size, representation_size,
                                    hidden_activation=nn.ELU())

    def forward(self, input, lambdas2):
        # input is (bs*K, representation_size)
        K = self.K
        rep_size = self.rep_size
        #actions_enc = self.action_encoder(actions)
        #input = torch.cat([input, actions_enc], -1)
        lambdas = input.view(-1, K, rep_size)
        bs = lambdas.shape[0]

        pairs = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                pairs.append(torch.cat([lambdas[:, i], lambdas[:, j]], -1))

        all_pairs = torch.stack(pairs, 1).view(bs*K,  K-1, -1)

        interaction = self.embedding_network(all_pairs)
        effect = self.effect_network(interaction)

        attention = self.attention_network(interaction)

        total_effect = (attention * effect).view(bs*K, (K-1), -1).sum(1) # TODO check this is right

        new_lambdas = self.encoder_network(total_effect)

        lambdas2 = self.encoder2_network(lambdas2)

        return new_lambdas, lambdas2

    def initialize_hidden(self, bs):
        return (Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))),
                Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))))