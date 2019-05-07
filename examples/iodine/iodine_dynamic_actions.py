from torch import nn

from rlkit.torch.iodine.iodine import IodineVAE

from rlkit.torch.conv_networks import BroadcastCNN
import rlkit.torch.iodine.iodine as iodine
from rlkit.torch.iodine.refinement_network import RefinementNetwork
from rlkit.torch.iodine.physics_network import PhysicsNetwork
from rlkit.torch.iodine.iodine_trainer import IodineTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.core import logger
import numpy as np
import h5py
from rlkit.torch.data_management.dataset import Dataset
from torch.utils.data.dataset import TensorDataset
import torch

def load_dataset(data_path, train=True, train_size=20):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features'])
    elif 'TwoBall' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features'])
        else:
            feats = np.array(hdf5_file['test']['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        return data
    elif 'BlocksGeneration' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features'])
            actions = np.array(hdf5_file['training']['actions'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
            actions = np.array(hdf5_file['validation']['actions'])

        t_sample = np.array([0, 2, 4, 6, 8, 10])
        feats = np.moveaxis(feats, -1, 2)[t_sample] # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1)[:10] # (bs, T, ch, imsize, imsize)
        actions = actions.squeeze()[:10] # (bs, action_dim)

        torch_dataset = TensorDataset(torch.Tensor(feats), torch.Tensor(actions))
        dataset = Dataset(torch_dataset)
        return dataset


def train_vae(variant):
    #train_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_train.hdf5'
    #test_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_test.hdf5'

    # train_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorBigTwoBallSmall.h5'
    # test_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorBigTwoBallSmall.h5'

    train_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock10kActions.h5'
    test_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock10kActions.h5'

    train_dataset = load_dataset(train_path, train=True)
    test_dataset = load_dataset(test_path, train=False)

    K = variant['vae_kwargs']['K']
    rep_size = variant['vae_kwargs']['representation_size']

    logger.get_snapshot_dir()
    variant['vae_kwargs']['architecture'] = iodine.imsize64_large_iodine_architecture
    variant['vae_kwargs']['decoder_class'] = BroadcastCNN

    refinement_net = RefinementNetwork(**iodine.imsize64_large_iodine_architecture['refine_args'],
                                       hidden_activation=nn.ELU())

    physics_net = PhysicsNetwork(K, rep_size, 13)
    m = IodineVAE(
        **variant['vae_kwargs'],
        refinement_net=refinement_net,
        dynamic=True,
        physics_net=physics_net,
    )

    m.to(ptu.device)

    t = IodineTrainer(train_dataset, test_dataset, m, variant['train_seedsteps'], variant['test_seedsteps'],
                       **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(epoch, save_vae=True, train=False, record_stats=True, batches=1,
                     save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, save_vae=False, train=True, record_stats=False, batches=1,
                     save_reconstruction=should_save_imgs)
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')


if __name__ == "__main__":
    variant = dict(
        vae_kwargs = dict(
            imsize=64,
            representation_size=128,
            input_channels=3,
            decoder_distribution='gaussian_identity_variance',
            beta=1,
            K=7,
            T=15,
            dataparallel=True
        ),
        algo_kwargs = dict(
            gamma=0.5,
            batch_size=8,
            lr=1e-4,
            log_interval=0,
        ),
        train_seedsteps=11,
        test_seedsteps=3,
        num_epochs=10000,
        algorithm='VAE',
        save_period=1,
        physics=True
    )


    run_experiment(
        train_vae,
        exp_prefix='iodine-blocks-physics-actions',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



