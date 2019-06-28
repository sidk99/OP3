import argparse
import h5py
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from base_dataset import BaseVideoDataset

import dataset_utils as du
from solid import process_bair_hard_obj_obs, pad

def read_cloth(args):
    sess = tf.Session()

    num_sims = {'train': 0, 'test': 0, 'val': 0}
    mode_map = {'training': 'train', 'validation': 'val'}
    hps = du.HyperParams(['num_frames', 'image_res', 'action_dim'])

    dataset = BaseVideoDataset(args.root, 1)
    for mode in ['train', 'test', 'val']:
        print('MODE: {}'.format(mode))
        img_tensor = dataset['images', mode]
        act_tensor = dataset['actions', mode]
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
        print('{} examples in mode {}'.format(counter, mode))
        num_sims[mode] += counter

    # redo just in case
    dataset = BaseVideoDataset(args.root, 1)
    iterators = {}
    for mode in ['train', 'test', 'val']:
        img_tensor = dataset['images', mode]
        act_tensor = dataset['actions', mode]
        iterators[mode] = {'obs': img_tensor, 'act': act_tensor}

    return num_sims, hps, iterators, mode_map


def write_cloth(iterators, mode, num_sims, num_frames, features_dataset, action_dataset):
    sess = tf.Session()
    counter = 0
    img_tensor = iterators[mode]['obs'] # this is a TF op
    act_tensor = iterators[mode]['act'] # this is a TF op
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
    print('{} examples in mode {}'.format(counter, mode))