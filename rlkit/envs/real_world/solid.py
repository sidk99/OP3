import argparse
import h5py
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from base_dataset import BaseVideoDataset

import dataset_utils as du

def read_solid(args):
    sess = tf.Session()

    num_sims = {'train': 0, 'test': 0, 'val': 0}
    mode_map = {'training': 'train', 'validation': 'val'}
    hps = du.HyperParams(['num_frames', 'image_res', 'action_dim'])

    for mode in ['train', 'test', 'val']:
        for partition in ['good', 'bad']:
            partition_path = os.path.join(args.root, partition)
            parition_dataset = BaseVideoDataset(partition_path, 1)
            print('MODE: {}'.format(mode))
            img_tensor = parition_dataset['images', mode]
            act_tensor = parition_dataset['actions', mode]
            counter = 0
            while True:
                try:
                    act_np = np.squeeze(sess.run(act_tensor), axis=0)  # (num_frames, 4)
                    obs_np = np.squeeze(sess.run(img_tensor), axis=0)  # (num_frames, 2, 48, 64, 3)
                    # print('\t{}: Actions: Shape: {} Norm: {}'.format(counter, act_np.shape, np.linalg.norm(act_np)))
                    # print('\t{}: Obs: Shape: {} Norm: {}'.format(counter, obs_np.shape, np.linalg.norm(obs_np)))
                    hps.update_hp('num_frames', obs_np.shape[0])
                    hps.update_hp('image_res', obs_np.shape[-2])
                    hps.update_hp('action_dim', act_np.shape[-1])

                    counter += 1
                except:
                    break
            print('{} examples in mode {} for partition {}'.format(counter, mode, partition))
            num_sims[mode] += counter

    # redo just in case
    iterators = {}
    for mode in ['train', 'test', 'val']:
        iterators[mode] = {}
        for partition in ['good', 'bad']:
            partition_path = os.path.join(args.root, partition)
            parition_dataset = BaseVideoDataset(partition_path, 1)
            img_tensor = parition_dataset['images', mode]
            act_tensor = parition_dataset['actions', mode]
            iterators[mode][partition] = {'obs': img_tensor, 'act': act_tensor}

    return num_sims, hps, iterators, mode_map


def write_solid(iterators, mode, num_sims, num_frames, features_dataset, action_dataset):
    sess = tf.Session()
    counter = 0
    for partition in ['good', 'bad']:
        img_tensor = iterators[mode][partition]['obs'] # this is a TF op
        act_tensor = iterators[mode][partition]['act'] # this is a TF op
        while True:
            try:
                act_np = np.squeeze(sess.run(act_tensor), axis=0)[:-1]  # (num_frames-1, 4)
                obs_np = np.squeeze(sess.run(img_tensor), axis=0)  # (num_frames, 2, 48, 64, 3)
                obs_np = process_bair_hard_obj_obs(obs_np)
                # print('\t{}: Actions: Shape: {} Norm: {}'.format(counter, act_np.shape, np.linalg.norm(act_np)))
                # print('\t{}: Obs: Shape: {} Norm: {}'.format(counter, obs_np.shape, np.linalg.norm(obs_np)))
                features_dataset[:, counter, :, :, :] = obs_np  # (num_frames, 64, 64, 3)
                action_dataset[:, counter, :] = act_np  # (num_frames, 4)
                counter += 1
            # assert False
            except:
                break
        print('{} examples in mode {} for partition {}'.format(counter, mode, partition))


def convert_data(args, reader, writer):
    fname = os.path.join(args.root, args.filename)
    print("Reading data")
    num_sims, hps, iterators, mode_map  = reader(args)
    print(num_sims)
    print(hps.hp)
    print("Writing data")
    num_frames = hps.get('num_frames')
    image_res = hps.get('image_res')
    action_dim = hps.get('action_dim')

    datasets = {'training':num_sims['train'],'validation':num_sims['val']}
    with h5py.File(fname, 'w') as f:
        for folder in datasets:
            num_sims = datasets[folder]
            cur_folder = f.create_group(folder)
            image_data_shape = (num_frames, num_sims, image_res, image_res, 3)
            action_data_shape = (num_frames, num_sims, action_dim)
            features_dataset = cur_folder.create_dataset('features', image_data_shape, dtype='uint8')
            action_dataset = cur_folder.create_dataset('actions', action_data_shape, dtype='float32')
            mode = mode_map[folder]

            writer(iterators, mode, num_sims, num_frames, features_dataset, action_dataset)

    print('Finished writing to {}'.format(fname))


def process_bair_hard_obj_obs(obs_np):
    """
        input
            obs_np: (num_frames, 2, 48, 64, 3)

        output:
            obs_np: (num_frames, 64, 64, 3)
    """
    # pad
    obs_np = pad(obs_np)  # (num_frames, 2, 64, 64, 3)
    # take only one view
    obs_np = obs_np [:, 1] # (num_frames, 64, 64, 3)
    return obs_np


def pad(obs_np):
    """
        input: (T, 2, 48, 64, 3)
        output: (T, 2, 64, 64, 3)
    """
    zeros = np.zeros_like(obs_np)[:, :, :8]
    padded = np.concatenate((zeros, obs_np, zeros), axis=2)
    return padded

