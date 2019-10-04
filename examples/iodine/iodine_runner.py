import time
import shutil

# from rlkit.core import logger

# from rlkit.torch.iodine.iodine import IodineVAE
#
# import rlkit.torch.iodine.iodine as iodine
# from rlkit.torch.iodine.iodine_trainer import IodineTrainer
# import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment

import rlkit.torch.iodine.iodine_v2 as iodine_v2
from rlkit.torch.iodine.iodine_trainer_v2 import TrainingScheduler, IodineTrainer


import numpy as np
import h5py
from rlkit.torch.data_management.dataset import Dataset, BlocksDataset
from torch.utils.data.dataset import TensorDataset
import torch
import random
from argparse import ArgumentParser
from rlkit.util.misc import get_module_path
import pdb

#-f twoBalls.h5 -n 2 -r 7 -c 0 -ns 1000 -nf 21
# class MyDataParallel(torch.nn.DataParallel):
#     def __getattr__(self, name):
#         print(name)
#         return getattr(self.module, name)


def load_dataset(data_path, train=True, size=None, batchsize=8, static=True):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        feats = np.array(hdf5_file['features'])[:size] #(B, 3, 84, 84)
        # pdb.set_trace()
        feats = np.expand_dims(feats, 0) #(1,B,3,84,84)
        torch_dataset = TensorDataset(torch.Tensor(feats))
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        T = feats.shape[1]
        print(feats.shape, np.max(feats), np.min(feats))
        return dataset, T
    elif 'twoBalls' in data_path:
        # if train:
        #     feats = np.array(hdf5_file['training']['features'])
        # else:
        #     feats = np.array(hdf5_file['test']['features'])
        # feats = feats.reshape((-1, 64, 64, 3))
        # feats = (feats * 255).astype(np.uint8)
        # feats = np.swapaxes(feats, 1, 3)
        # T = feats.shape[1]
        # print(feats.shape, np.max(feats), np.min(feats))
        if train:
            feats = np.array(hdf5_file['training']['features']) #(51,1000,64,64,3), values btwn 0-1
        else:
            feats = np.array(hdf5_file['validation']['features'])
        # pdb.set_trace()
        feats = np.transpose(feats, (1, 0, 4, 2, 3))[:size, :8] * 255 #(B,8,3,64,46)
        torch_dataset = TensorDataset(torch.Tensor(feats))
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        T = feats.shape[1]
        print(feats.shape, np.max(feats), np.min(feats))
        return dataset, T
    elif 'stack' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features']) # (T, bs, ch, imsize, imsize)
            actions = np.array(hdf5_file['training']['actions'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
            actions = np.array(hdf5_file['validation']['actions'])
        #t_sample = [0, 2, 4, 6, 9]
        # pdb.set_trace()
        feats = np.moveaxis(feats, -1, 2) #[t_sample] # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1) # (bs, T, ch, imsize, imsize)
        #actions = np.moveaxis(actions, 0, 1) # (bs, T, action_dim)
        torch_dataset = TensorDataset(torch.Tensor(feats)[:size])
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
        T = feats.shape[1]
        print(feats.shape, np.max(feats), np.min(feats))
        return dataset, T
    # elif 'pickplace' in data_path:
    #     if train:
    #         feats = np.array(hdf5_file['training']['features'])
    #         actions = np.array(hdf5_file['training']['actions'])
    #     else:
    #         feats = np.array(hdf5_file['validation']['features'])
    #         actions = np.array(hdf5_file['validation']['actions'])
    #
    #     feats = np.moveaxis(feats, -1, 2) # (T, bs, ch, imsize, imsize)
    #     feats = np.moveaxis(feats, 0, 1) # (bs, T, ch, imsize, imsize)
    #     actions = np.moveaxis(actions, 0, 1) # (bs, T, action_dim)
    #     torch_dataset = TensorDataset(torch.Tensor(feats[:size]), torch.Tensor(actions[:size]))
    #     dataset = BlocksDataset(torch_dataset, batchsize=batchsize)
    #     T = feats.shape[1]
    #     print(feats.shape, np.max(feats), np.min(feats))
    #     return dataset, T
    elif 'cloth' in data_path or 'poke' in data_path or 'solid' in data_path \
            or 'kevin' in data_path or 'pickplace' in data_path:
        #cloth: bs=13866, T=20, action_dim=4
        #poke: bs=5425, T=20, action_dim=5
        #solid: bs=7143, T=30, action_dim=4
        #kevin: bs=1500, T=15, action_dim=2
        #pickplace_multienv_10k: bs=10000, T=21, action_dim=6
        if train:
            feats = np.array(hdf5_file['training']['features'])
            actions = np.array(hdf5_file['training']['actions'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
            actions = np.array(hdf5_file['validation']['actions'])

        feats = np.moveaxis(feats, -1, 2)  # (T, bs, ch, imsize, imsize)
        feats = np.moveaxis(feats, 0, 1)  # (bs, T, ch, imsize, imsize)
        actions = np.moveaxis(actions, 0, 1)  # (bs, T-1, action_dim) EXCEPT for pickplace envs which are (bs,T,A) instead
        if static:
            bs, T = feats.shape[0], feats.shape[1]
            if size == None:
                size = bs
            rand_ts = np.random.randint(0, T, size=size) #As the first timesteps could be correlated
            # pdb.set_trace()
            tmp = torch.Tensor(feats[range(size), rand_ts]).unsqueeze(1) #(size, 1, ch, imsize, imsize)
            torch_dataset = TensorDataset(tmp)
        else:
            torch_dataset = TensorDataset(torch.Tensor(feats[:size]), torch.Tensor(actions[:size]))
        dataset = BlocksDataset(torch_dataset, batchsize=batchsize)

        if 'pickplace' in data_path:
            dataset.action_dim = 4 #Changing it from 6

        T = feats.shape[1]
        print(feats.shape, np.max(feats), np.min(feats))
        # pdb.set_trace()
        return dataset, T
    else:
        raise ValueError("Invalid dataset given: {}".format(data_path))


#Dataset information regarding T
# dataset_info = {
#     'pickplace_1env_1k.h5': 21
#     'pickplace_multienv_10k.h5': 21
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

def copy_to_save_file(dir_str):
    base = get_module_path()
    # shutil.copy2(base+'/rlkit/torch/iodine/iodine_runner.py', dir_str)
    # shutil.copy2(base+'/rlkit/torch/iodine/iodine_trainer_v2.py', dir_str)
    # shutil.copy2(base+'/rlkit/torch/iodine/physics_network_v2.py', dir_str)
    # shutil.copy2(base+'/rlkit/torch/iodine/refinement_network_v2.py', dir_str)
    shutil.copytree(base + '/rlkit/torch/iodine', dir_str+'/saved_torch_iodine_files')
    shutil.copy2(base + '/examples/iodine/iodine_runner.py', dir_str + '/saved_iodine_runner.py')

def train_vae(variant):
    from rlkit.core import logger
    copy_to_save_file(logger.get_snapshot_dir())
    # seed = 1
    seed = int(variant['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ######Dataset loading######
    train_path = get_module_path() + '/ec2_data/{}.h5'.format(variant['dataset'])  #ec2_data, data
    test_path = train_path
    bs = variant['training_args']['batch_size']
    train_size = 100 if variant['debug'] == 1 else None

    static = (variant['schedule_args']['schedule_type'] == 'static_iodine') #Boolean
    train_dataset, max_T = load_dataset(train_path, train=True, batchsize=bs, size=train_size, static=static)
    test_dataset, _ = load_dataset(test_path, train=False, batchsize=bs, size=100, static=static)
    print(logger.get_snapshot_dir())

    ######Model loading######
    op3_args = variant["op3_args"]
    m = iodine_v2.create_model_v2(op3_args, op3_args['det_repsize'], op3_args['sto_repsize'], action_dim=train_dataset.action_dim)
    if variant['dataparallel']:
        # m = MyDataParallel(m)
        m = torch.nn.DataParallel(m)
    m.cuda()

    ######Training######
    scheduler = TrainingScheduler(**variant["schedule_args"], max_T = max_T)
    t = IodineTrainer(train_dataset, test_dataset, m, scheduler, **variant["training_args"])

    save_period = variant['save_period']
    best_loss, num_stalled = 1e6, 0
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)

        # t0 = time.time()
        train_stats = t.train_epoch(epoch)
        # t1 = time.time()
        # print(t1-t0)
        test_stats = t.test_epoch(epoch, train=False, batches=1, save_reconstruction=should_save_imgs)
        t.test_epoch(epoch, train=True, batches=1, save_reconstruction=should_save_imgs)
        for k, v in {**train_stats, **test_stats}.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        t.save_model()

        #Early termination if stalling for 5 epochs
        # if train_stats["train/loss"] < best_loss:
        #     best_loss = train_stats["train/loss"]
        #     num_stalled = 0
        # else:
        #     num_stalled += 1
        #     if num_stalled == 5:
        #         break

        # torch.save(m.state_dict(), open(logger.get_snapshot_dir() + '/params.pkl', "wb"))
    # logger.save_extra_data(m, 'vae.pkl', mode='pickle')

# Datasets: pickplace_1env_1k, pickplace_multienv_10k, stack_o2p2_60k, cloth, poke, solid, twoBalls, twoBalls_10k,
#   pickplace_1block_10k, pickplace_o12_v2_10k, pickplace_o12_noise_10k
# Generic run: CUDA_VISIBLE_DEVICES=[A,B,C...] python iodine_runner.py -de [0/1] -da [DATASET_NAME_HERE]
#   -da options: look at above list of Datasets
#   -de options: 0 for training on full dataset, 1 for training on first 100 sequences
# Example run: CUDA_VISIBLE_DEVICES=1 python iodine_runner.py -de 1 -da stack_o2p2_60k


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-da', '--dataset', type=str, default=None, required=True)
    parser.add_argument('-de', '--debug', type=int, default=1)
    parser.add_argument('-m', '--mode', type=str,default='here_no_doodad')

    args = parser.parse_args()

    variant = dict(
        op3_args = dict(
            refinement_model_type = "size_dependent_conv",  # size_dependent_conv, size_dependent_conv_no_share
            decoder_model_type = "reg",  # reg, reg_no_share
            dynamics_model_type = "reg_ac32",  # reg_ac32, reg_ac32_no_share
            sto_repsize = 64,
            det_repsize = 64,
            extra_args = dict(
                beta = 1e-2,
                deterministic_sampling = False
            ),
            K=4
        ),
        schedule_args = dict(  # Arguments for TrainingScheduler
            seed_steps = 4,
            T = 5,  # Max number of steps into the future we want to go or max length of a schedule
            schedule_type = 'curriculum',  # single_step_physics, curriculum, static_iodine, rprp, next_step, random_alternating
        ),
        training_args = dict(  # Arguments for IodineTrainer
            batch_size= 10,  # Change to appropriate constant based off dataset size
            lr=3e-4,
        ),
        num_epochs = 300,
        save_period=2,
        dataparallel=True,
        dataset=args.dataset,
        debug=args.debug,
        machine_type='g3.16xlarge'  # Note: Purely for logging purposed and NOT used for setting actual machine type
    )

    # Relevant options: 'here_no_doodad', 'local_docker', 'ec2'
    run_experiment(
        train_vae,
        exp_prefix='{}-{}-v2-reg'.format(args.dataset, variant['schedule_args']['schedule_type']),
        mode=args.mode,
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=None,
        region='us-west-2'
    )





