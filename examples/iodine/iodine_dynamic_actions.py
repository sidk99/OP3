from torch import nn

from rlkit.torch.iodine.iodine import IodineVAE

import rlkit.torch.iodine.iodine as iodine
from rlkit.torch.iodine.iodine_trainer import IodineTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.core import logger
import numpy as np
import h5py
from rlkit.torch.data_management.dataset import Dataset, BlocksDataset
from torch.utils.data.dataset import TensorDataset
import torch
import random

def load_dataset(data_path, train=True, size=None, batchsize=8):
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
    elif 'pickplace' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features'])
            actions = np.array(hdf5_file['training']['actions'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
            actions = np.array(hdf5_file['validation']['actions'])

        feats = np.moveaxis(feats, -1, 2) # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1) # (bs, T, ch, imsize, imsize)
        feats = (feats * 255).astype(np.uint8)
        actions = np.moveaxis(actions, 0, 1) # (bs, T, action_dim)

        torch_dataset = TensorDataset(torch.Tensor(feats)[:size],
                                      torch.Tensor(actions)[:size])
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        return dataset




def train_vae(variant):
    #train_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_train.hdf5'
    #test_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_test.hdf5'

    # train_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorBigTwoBallSmall.h5'

    #train_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock10kActions.h5'
    seed = int(variant['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    train_path = '/home/jcoreyes/objects/rlkit/data/pickplace1k.h5'
    test_path = train_path
    bs = variant['algo_kwargs']['batch_size']
    train_dataset = load_dataset(train_path, train=True, batchsize=bs, size=None)
    test_dataset = load_dataset(test_path, train=False, batchsize=bs, size=10)

    logger.get_snapshot_dir()

    m = iodine.create_model(variant['model'], train_dataset.action_dim, dataparallel=False)
    if variant['dataparallel']:
        m = torch.nn.DataParallel(m)

    #m.to(ptu.device)

    t = IodineTrainer(train_dataset, test_dataset, m,
                       **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(epoch, save_vae=False, train=False, record_stats=True, batches=1,
                     save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, save_vae=False, train=True, record_stats=False, batches=1,
                     save_reconstruction=should_save_imgs)
        torch.save(m.state_dict(), open(logger._snapshot_dir + '/params.pkl', "wb"))
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')


if __name__ == "__main__":
    variant = dict(
        model=iodine.imsize64_large_iodine_architecture,
        algo_kwargs = dict(
            gamma=0.5,
            batch_size=8,
            lr=1e-4,
            log_interval=0,
            train_T=5,
            test_T=5,
            seed_steps=4,
        ),
        num_epochs=10000,
        algorithm='Iodine',
        save_period=1,
        dataparallel=True,
    )


    run_experiment(
        train_vae,
        exp_prefix='iodine-blocks-pickplace',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=100,
    )



