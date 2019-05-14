from rlkit.torch.vae.ray_image_logger import ImageLogger
from torch import nn

from rlkit.torch.iodine.iodine import IodineVAE

import rlkit.torch.iodine.iodine as iodine
from rlkit.torch.iodine.iodine_trainer import IodineTrainer
import ray
import ray.tune as tune
from rlkit.torch.vae.ray_vae_trainer import RayVAETrainer
from rlkit.launchers.ray.launcher import launch_experiment
from rlkit.core import logger
import numpy as np
import h5py
from rlkit.torch.data_management.dataset import Dataset, BlocksDataset
from torch.utils.data.dataset import TensorDataset
import torch
import random
from argparse import ArgumentParser

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
    elif 'stack' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features'])
            actions = np.array(hdf5_file['training']['actions'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
            actions = np.array(hdf5_file['validation']['actions'])
        t_sample = [0, 2, 4, 6, 9]
        feats = np.moveaxis(feats, -1, 2)[t_sample] # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1) # (bs, T, ch, imsize, imsize)
        #feats = (feats * 255).astype(np.uint8)
        #actions = np.moveaxis(actions, 0, 1) # (bs, T, action_dim)
        torch_dataset = TensorDataset(torch.Tensor(feats)[:size])
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)

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
        #feats = (feats * 255).astype(np.uint8)
        actions = np.moveaxis(actions, 0, 1) # (bs, T, action_dim)

        torch_dataset = TensorDataset(torch.Tensor(feats)[:size],
                                      torch.Tensor(actions)[:size])
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        return dataset


def run_experiment_func(variant):
    #train_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_train.hdf5'
    #test_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_test.hdf5'

    # train_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorBigTwoBallSmall.h5'

    #train_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock10kActions.h5'
    # seed = int(variant['seed'])
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)


    train_path = '/home/ubuntu/objects/rlkit/data/%s.h5' % variant['dataset']
    test_path = train_path
    bs = variant['algo_kwargs']['batch_size']
    train_size = 4 if variant['debug'] == 1 else None
    train_dataset = load_dataset(train_path, train=True, batchsize=bs, size=train_size)
    test_dataset = load_dataset(test_path, train=False, batchsize=bs, size=100)

    logger.get_snapshot_dir()

    m = iodine.create_model(variant['model'], train_dataset.action_dim, dataparallel=False)
    # if variant['dataparallel']:
    #     m = torch.nn.DataParallel(m)
    #
    # #m.to(ptu.device)
    #
    t = IodineTrainer(train_dataset, test_dataset, m,
                       **variant['algo_kwargs'])
    # save_period = variant['save_period']
    # for epoch in range(variant['num_epochs']):
    #     should_save_imgs = (epoch % save_period == 0)
    #     t.train_epoch(epoch)
    #     t.test_epoch(epoch, save_vae=False, train=False, record_stats=True, batches=1,
    #                  save_reconstruction=should_save_imgs)
    #     t.test_epoch(epoch, save_vae=False, train=True, record_stats=False, batches=1,
    #                  save_reconstruction=should_save_imgs)
    #     torch.save(m.state_dict(), open(logger._snapshot_dir + '/params.pkl', "wb"))
    # logger.save_extra_data(m, 'vae.pkl', mode='pickle')

    algo = RayVAETrainer(t, train_dataset, test_dataset, variant)

    return algo


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-da', '--dataset', type=str, default=None) # stack50k
    parser.add_argument('-de', '--debug', type=int, default=1)

    args = parser.parse_args()

    variant = dict(
        model=iodine.imsize64_large_iodine_architecture,
        algo_kwargs = dict(
            gamma=0.5,
            batch_size=2,
            lr=1e-4,
            log_interval=0,
            train_T=5,  # 15
            test_T=5,   # 9
            seed_steps=4,
        ),
        num_epochs=10000,
        algorithm='Iodine',
        save_period=1,
        dataparallel=False,
        dataset=args.dataset,
        debug=args.debug
    )

    n_seeds = 1
    mode = 'aws' #'local_docker'
    exp_prefix = 'iodine-blocks-%s' % args.dataset

    launch_experiment(
        mode=mode,
        use_gpu=False,

        local_launch_variant=dict(
            seeds=n_seeds,
            init_algo_functions_and_log_fnames=[(run_experiment_func, 'progress.csv')],
            exp_variant=variant,
            checkpoint_freq=20,
            exp_prefix=exp_prefix,
            #custom_loggers=[ImageLogger],
            # resources_per_trial={
            #     'cpu': 2,
            # }
        ),
        remote_launch_variant=dict(
            # head_instance_type='m1.xlarge',
            max_spot_price=.2,
        ),
        docker_variant=dict(),
    )




