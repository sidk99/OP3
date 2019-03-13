from torch import nn

from rlkit.torch.vae.iodine import IodineVAE

from rlkit.torch.conv_networks import BroadcastCNN
import rlkit.torch.vae.iodine as iodine
from rlkit.torch.vae.refinement_network import RefinementNetwork
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
import rlkit.torch.vae.conv_vae as conv_vae
from rlkit.torch.vae.unet import UNet
from rlkit.torch.vae.iodine_trainer import IodineTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.pythonplusplus import identity
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.core import logger
import numpy as np
from scipy import misc
import h5py

def load_dataset(data_path, train=True):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features'])
    else:
        if train:
            feats = np.array(hdf5_file['training']['features'])
        else:
            feats = np.array(hdf5_file['test']['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        return data


def train_vae(variant):
    train_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_train.hdf5'
    test_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_test.hdf5'

    #train_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorTwoBallSmall.h5'
    #test_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorTwoBallSmall.h5'

    train_data = load_dataset(train_path, train=True)
    test_data = load_dataset(test_path, train=False)

    train_data = train_data.reshape((train_data.shape[0], -1))[:500]
    test_data = test_data.reshape((test_data.shape[0], -1))[:100]
    #logger.save_extra_data(info)
    logger.get_snapshot_dir()
    variant['vae_kwargs']['architecture'] = iodine.imsize84_iodine_architecture
    variant['vae_kwargs']['decoder_class'] = BroadcastCNN

    refinement_net = RefinementNetwork(**iodine.imsize84_iodine_architecture['refine_args'],
                                       hidden_activation=nn.ELU())
    m = IodineVAE(
        **variant['vae_kwargs'],
        refinement_net=refinement_net

    )

    m.to(ptu.device)
    t = IodineTrainer(train_data, test_data, m,
                       **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')


if __name__ == "__main__":
    variant = dict(
        vae_kwargs = dict(
            imsize=84,
            representation_size=128,
            input_channels=3,
            decoder_distribution='gaussian_identity_variance',
            beta=0.1,
        ),
        algo_kwargs = dict(
            gamma=0.5,
            batch_size=4,
            lr=3e-4,
            log_interval=0,
        ),
        num_epochs=2000,
        algorithm='VAE',
        save_period=5,
    )


    run_experiment(
        train_vae,
        exp_prefix='iodine-clevr',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



