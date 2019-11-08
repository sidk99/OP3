from __future__ import print_function
import sys
sys.path.insert(0, '/media/sidk/Data/sidk/Research/OP3/')
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import h5py
import numpy as np
from livelossplot import PlotLosses
from rlkit.torch.vae.vae_base import GaussianLatentVAE


np.random.seed(0)


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--debug', default=False,
                    help='debug flag')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(1)

device = torch.device("cuda" if args.cuda else "cpu")

def load_dataset(data_path):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file

    return hdf5_file


def visualize_parameters(model):
    for n, p in model.named_parameters():
        if p.grad is None:
            print('{}\t{}\t{}'.format(n, p.data.norm(), None))
        else:
            print('{}\t{}\t{}'.format(n, p.data.norm(), p.grad.data.norm()))


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


train_path = '/home/sidk/Module-Transform/dataholder'
test_path = '/home/sidk/Module-Transform/dataholder'

dataloader = load_dataset(train_path)
train_data_size = 9000
test_data_size=1000

def get_batch( train, batch_size):
    if train:
        ind = np.random.randint(0,train_data_size-batch_size)
        samples = dataloader['/train/0/[]/input'][ind:ind+batch_size]
    else:
        ind = np.random.randint(0, test_data_size-batch_size)
        samples = dataloader['/test/0/[]/input'][ind:ind+batch_size]
    return torch.Tensor(samples)

class VAE(GaussianLatentVAE):
    def __init__(self, representation_size=0):
        #super(VAE, self).__init__()
        super().__init__(representation_size)

        self.fc1 = nn.Linear(4, 64)
        self.fc21 = nn.Linear(64, 32)
        self.fc22 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 40)

        visualize_parameters(self)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    #def reparameterize(self, mu, logvar):
    #    std = torch.exp(0.5*logvar)
    #    eps = torch.randn_like(std)
    #    return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        mid2 = self.fc4(h3).view(args.batch_size*4,10)
        return F.softmax(mid2, dim=-1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize((mu, logvar))
        return self.decode(z), mu, logvar

    def logprob(self, inputs, obs_distribution_params):
        pass







# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta=1):
    #print(x.shape, "  ", recon_x.shape)
    #BCE = F.nll_loss(recon_x.view(args.batch_size*4 , 10), x.view(args.batch_size*4).long(), reduction='sum')
    BCE = F.cross_entropy(recon_x, x.view(args.batch_size*4).long(), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = beta*(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

    return BCE + KLD, BCE, KLD



def train(epoch, plotter, beta=1, prefix=''):
    model.train()
    train_loss = 0
    train_acc=0
    count=0
    logs = {}#{'total loss': [], 'kl loss':[] , 'crossentropy loss': []}
    for i in range(2):#int(train_data_size/args.batch_size)):  #batch_idx, (data, _) in enumerate(train_loader):
        data = get_batch(True, args.batch_size).to(device)
        #print(data, " , ")
        #if i % 15 == 0:
        #    visualize_parameters(model)
        print("Parameters at Start:   ")
        visualize_parameters(model)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, crossentr_loss, kl_loss = loss_function(recon_batch, data, mu, logvar, beta)
        print("Parameters after forward pass:   ")
        visualize_parameters(model)

        loss.backward()
        train_loss += loss.item()
        print("Parameters after loss backward:   ")
        visualize_parameters(model)

        optimizer.step()
        print("Parameters after optimizer step:   ")
        visualize_parameters(model)

        extractval = recon_batch.view(args.batch_size, 4, 10)
        _, idx = torch.max(extractval, -1)
        idx = idx.squeeze().type(torch.FloatTensor)
        #output = output.squeeze().type(torch.LongTensor)

        for j in range(idx.shape[0]):
            count+=1
            if (torch.equal(idx[j].to(device), data[j].to(device))):
                train_acc += 1


        logs['total loss'] = loss.item()/args.batch_size
        logs['kl loss'] = (1/beta)*(kl_loss.item()/args.batch_size)
        logs['crossentropy loss'] = crossentr_loss.item()/args.batch_size
        logs['train accuracy loss with beta: '+ str(beta)] = train_acc / count
        plotter.update(logs)
        #liveloss.draw()
    print("COUNT:  ", count, "TRAIN DATA SIZE:  ", train_data_size)

    #print('This is output:  ', idx, '  This is yval:  ', data)
    print("Training Accuracy: ", train_acc/count)
    #print("shape of odx : ", idx.shape, "shape of data: ", data.shape)

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KLE  Loss: {:.6f}, Cross entropy Loss {:.6f}'.format(
        epoch, i * args.batch_size, train_data_size,
               100. * i / train_data_size,
               loss.item() / args.batch_size, kl_loss.item() / args.batch_size,
               crossentr_loss.item() / args.batch_size))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / train_data_size))


def test(epoch, plotter=None):
    model.eval()
    test_loss = 0
    test_acc = 0
    logs={}
    count=0
    with torch.no_grad():
        for i in range(int(test_data_size/args.batch_size)): #for i, (data, _) in enumerate(test_loader):
            data = get_batch(False, args.batch_size).to(device)
            #print("TEST:", data.shape)
            recon_batch, mu, logvar = model(data)
            test_loss_temp, crossentropy_loss, kl_loss = loss_function(recon_batch, data, mu, logvar)
            test_loss+= test_loss_temp

            extractval = recon_batch.view(args.batch_size, 4, 10)
            _, idx = torch.max(extractval, -1)
            idx = idx.squeeze().type(torch.FloatTensor)
            # output = output.squeeze().type(torch.LongTensor)

            for j in range(idx.shape[0]):
                count+=1
                if (torch.equal(idx[j].to(device), data[j].to(device))):
                    test_acc += 1
            logs['test accuracy loss with beta: ' + str(beta)] = test_acc / count
            plotter.update(logs)
            #if i == 0:
              #  n = min(data.size(0), 8)
                #comparison = torch.cat([data[:n],
                #                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                #save_image(comparison.cpu(),
                #         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= test_data_size
    print('====> Test set loss: {:.4f}  , Cross Entropy: {:.4f},   KL: {:.4f}'.format(test_loss.item(), crossentropy_loss.item(), kl_loss.item()) )

if __name__ == "__main__":

    for i in range(2,3):
        traindata = PlotLosses(fig_path='data/train with beta: '+str(i)+'.png')
        testdata = PlotLosses(fig_path='data/test with beta: '+str(i)+'.png')
        np.random.seed(0)
        torch.manual_seed(1)
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        beta = pow(10,-i)
        print(beta)
        num_epochs =args.epochs
        if args.debug:
            num_epochs=1
        for epoch in range(1, num_epochs + 1):
            print( "Epoch: ", str(epoch))
            train(epoch, traindata, beta, prefix=str(i)+' ')
            test(epoch, plotter=testdata)
        traindata.draw()
        testdata.draw()

        #with torch.no_grad():
        #    sample = torch.randn(64, 32).to(device)
        #    sample = model.decode(sample).cpu()
        #    save_image(sample.view(64, 1, 28, 28),
         #              'results/sample_' + str(epoch) + '.png')