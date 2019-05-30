from torch import nn
import time

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
from argparse import ArgumentParser
from rlkit.util.misc import get_module_path
import pdb

def load_dataset(data_path, train=True, size=None, batchsize=8):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features']), None
    elif 'TwoBall' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features'])
        else:
            feats = np.array(hdf5_file['test']['features'])
        feats = feats.reshape((-1, 64, 64, 3))
        feats = (feats * 255).astype(np.uint8)
        feats = np.swapaxes(feats, 1, 3)
        T = feats.shape[1]
        print(feats.shape, np.max(feats), np.min(feats))
        return feats, T
    elif 'stack' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features']) # (T, bs, ch, imsize, imsize)
            actions = np.array(hdf5_file['training']['actions'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
            actions = np.array(hdf5_file['validation']['actions'])
        #t_sample = [0, 2, 4, 6, 9]
        feats = np.moveaxis(feats, -1, 2) #[t_sample] # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1) # (bs, T, ch, imsize, imsize)
        #actions = np.moveaxis(actions, 0, 1) # (bs, T, action_dim)
        torch_dataset = TensorDataset(torch.Tensor(feats)[:size])
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        T = feats.shape[1]
        print(feats.shape, np.max(feats), np.min(feats))
        return dataset, T
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
        torch_dataset = TensorDataset(torch.Tensor(feats[:size]),
                                      torch.Tensor(actions[:size]))
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        T = feats.shape[1]
        print(feats.shape, np.max(feats), np.min(feats))
        return dataset, T

#Dataset information regarding T
# dataset_info = {
#     'pickplace_1env_1k.h5': 21
# }

# (10000, 21, 3, 64, 64) 182 0
# (100, 21, 3, 64, 64) 186 0

# (1000, 2, 3, 64, 64) 160 0
# (100, 2, 3, 64, 64) 160 0

# train_dataset,
#             test_dataset,
#             model,
#             train_T=5,
#             test_T=5,
#             max_T=None,
#             seed_steps=4,
#             schedule_type='single_step_physics',
#             batch_size=128,
#             log_interval=0,
#             gamma=0.5,
#             lr=1e-3,



def train_vae(variant):
    # print("HIHIHIHIHI")
    seed = int(variant['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    variant['model']['schedule_kwargs'] = variant['schedule_kwargs'] #Adding it to dictionary
    variant['model']['K'] = variant['K'] #Adding K to model dictionary

    train_path = get_module_path() + '/ec2_data/%s.h5' % variant['dataset']
    test_path = train_path
    bs = variant['training_kwargs']['batch_size']
    train_size = 100 if variant['debug'] == 1 else None
    train_dataset, max_T = load_dataset(train_path, train=True, batchsize=bs, size=train_size)
    test_dataset, _ = load_dataset(test_path, train=False, batchsize=bs, size=100)

    print(logger.get_snapshot_dir())
    # pdb.set_trace()

    m = iodine.create_model(variant['model'], train_dataset.action_dim)
    if variant['dataparallel']:
        m = torch.nn.DataParallel(m)
    m.cuda()

    t = IodineTrainer(train_dataset, test_dataset, m,
                       **variant['training_kwargs'], **variant['schedule_kwargs'], max_T=max_T)
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)

        t0 = time.time()
        train_stats = t.train_epoch(epoch)
        t1 = time.time()
        print(t1-t0)
        test_stats = t.test_epoch(epoch, train=False, batches=1,save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, train=True, batches=1, save_reconstruction=should_save_imgs)
        for k, v in {**train_stats, **test_stats}.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()

        torch.save(m.state_dict(), open(logger.get_snapshot_dir() + '/params.pkl', "wb"))
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')

#CUDA_VISIBLE_DEVICES=1,2 python iodine.py -da pickplace_1env_1k -de 0
#CUDA_VISIBLE_DEVICES=1 python iodine.py -da pickplace_multienv_10k -de 0
#CUDA_VISIBLE_DEVICES=1,2 python iodine.py -da stack_o2p2_60k -de 0
#CUDA_VISIBLE_DEVICES=1,2 python iodine.py -da pickplace_multienv_c3_10k -de 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-da', '--dataset', type=str, default=None, required=True) # stack50k, pickplace_1env_1k
    parser.add_argument('-de', '--debug', type=int, default=1)
    parser.add_argument('-m', '--mode', type=str,default='here_no_doodad')

    args = parser.parse_args()

    variant = dict(
        model=iodine.imsize64_large_iodine_architecture_multistep_physics,   #imsize64_small_iodine_architecture,   #imsize64_large_iodine_architecture_multistep_physics,
        K=7,
        training_kwargs = dict(
            batch_size=1, #Used in IodineTrainer, change to appropriate constant based off dataset size
            lr=1e-4, #Used in IodineTrainer, sweep
            log_interval=0,
        ),
        schedule_kwargs=dict(
            train_T=5, #Number of steps in single training sequence, change with dataset
            test_T=5,  #Number of steps in single testing sequence, change with dataset
            seed_steps=4, #Number of seed steps
            schedule_type='single_step_physics' #single_step_physics, single_step_physics
        ),
        num_epochs=5,
        algorithm='Iodine',
        save_period=1,
        dataparallel=True,
        dataset=args.dataset,
        debug=args.debug
    )

    #Relevant options: 'here_no_doodad', 'local_docker', 'ec2'
    run_experiment(
        train_vae,
        exp_prefix='iodine-blocks-%s' % args.dataset,
        mode=args.mode,
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=None,
        region='us-west-2'
    )



