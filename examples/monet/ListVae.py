import sys
sys.path.insert(0, '/media/sidk/Data/sidk/Research/OP3/')
#import rlkit
from rlkit.torch.monet.Listmodel import ListVAE
from transitionmain import VAE
#from rlkit.torch.conv_networks import BroadcastCNN
#import rlkit.torch.monet.monet as monet
#from rlkit.torch.monet.unet import UNet
#from rlkit.torch.monet.monet_trainer import MonetTrainer
from rlkit.torch.monet.list_trainer import ListTrainer

import rlkit.torch.pytorch_util as ptu
#from rlkit.pythonplusplus import identity
from rlkit.launchers.launcher_util import run_experiment
from rlkit.core import logger
import numpy as np
import h5py
import torch

np.random.seed(0)
#torch.manual_seed(1)

def load_dataset(data_path, train=True):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features'])
    else:
        if train:
            feats = np.array(hdf5_file['train/0/[]/input'])
        else:
            feats = np.array(hdf5_file['test/0/[]/input'])
        #data = feats.reshape((-1, 64, 64, 3))
        #data = (data * 255).astype(np.uint8)
        #data = np.swapaxes(data, 1, 3)
        return feats




def train_vae(variant):
    #train_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_train_10000.hdf5'
    #test_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_test.hdf5'

    train_path = '/media/sidk/Data/sidk/Research/OP3/data/dataholder.hdf5'#/home/sidk/Desktop/research/Research/OP3/data/dataholder.hdf5'
    test_path = '/media/sidk/Data/sidk/Research/OP3/data/dataholder.hdf5'#/home/sidk/Desktop/research/Research/OP3/data/dataholder/hdf5'

    train_data = load_dataset(train_path, train=True)
    test_data = load_dataset(test_path, train=False)

    # train_data = train_data.reshape((train_data.shape[0], -1))
    # test_data = test_data.reshape((test_data.shape[0], -1))
    #logger.save_extra_data(info)
    logger.get_snapshot_dir()


    #attention_net = UNet(in_channels=4, n_classes=1, up_mode='upsample', depth=3,
    #                     padding=True)
    np.random.seed(0)
    torch.manual_seed(1)
    m = ListVAE(
        variant['list_size'],
    )

    m.to(ptu.device)
    t = ListTrainer(train_data, test_data, m, representation_size=variant['representation_size'],
                    batch_size= variant['batch_size']
                       )
    #save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        #should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            #save_reconstruction=should_save_imgs,
        )
        #if should_save_imgs:
        #    t.dump_samples(epoch)
    #logger.save_extra_data(m, 'vae.pkl', mode='pickle')


if __name__ == "__main__":
    variant = dict(
        list_size=4,
        num_epochs=50,
        representation_size=64,
        batch_size = 128
        #algorithm='VAE',
        #save_period=5,
    )

    #train_vae(variant)
    run_experiment(
        train_vae,
        exp_prefix='vae-clevr',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



