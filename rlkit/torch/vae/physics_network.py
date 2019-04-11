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
    ):
        self.save_init_params(locals())
        super().__init__()
        self.K = K
        self.rep_size = representation_size

        enc_size = 128
        self.embedding_network = Mlp((128,), enc_size, representation_size*2, hidden_activation=nn.ELU(),
                                     output_activation=nn.ELU())
        self.effect_network = Mlp((128,), enc_size, enc_size, hidden_activation=nn.ELU(),
                                     output_activation=nn.ELU())
        self.attention_network = Mlp((128,), 1, enc_size, hidden_activation=nn.ELU(),
                                     output_activation=nn.Sigmoid())
        self.encoder_network = Mlp((128,), representation_size, enc_size, hidden_activation=nn.ELU())

    def forward(self, input):
        # input is (bs*K, representation_size)
        K = self.K
        rep_size = self.rep_size
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

        return new_lambdas

    def initialize_hidden(self, bs):
        return (Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))),
                Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))))