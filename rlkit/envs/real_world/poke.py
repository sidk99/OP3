import argparse
import cv2
import h5py
import glob
import math
import numpy as np
import os
import scipy.ndimage
from tqdm import tqdm

import dataset_utils as du

def read_poke(args):
    num_sims = {'train': 0, 'test': 0}
    hps = du.HyperParams(['num_frames', 'image_res', 'action_dim'])
    mode_map = {'training': 'train', 'validation': 'test'}
    iterators = {}
    for mode in ['train', 'test']:
        iterators[mode] = sorted(glob.glob("{}/*".format(os.path.join(args.root, mode))))
        counter = 0
        for run in tqdm(iterators[mode]):
            actions = np.load(run + "/actions.npy")
            imgs = sorted(glob.glob('{}/*.jpg'.format(run)))
            assert len(actions) == len(imgs)-1
            for i in range(0, len(actions), args.num_frames)[:-1]:
                obs_np = []
                for j in range(i, i+args.num_frames):
                    img = cv2.imread(imgs[j])
                    img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                    obs_np.append(img)
                obs_np = np.stack(obs_np)  # (num_frames, 64, 64, 3)
                act_np = actions[i:i+args.num_frames]  # (num_frames, 5)
                hps.update_hp('num_frames', obs_np.shape[0])
                hps.update_hp('image_res', obs_np.shape[-2])
                hps.update_hp('action_dim', act_np.shape[-1])
                counter += 1
        num_sims[mode] += counter
    return num_sims, hps, iterators, mode_map

def write_poke(iterators, mode, num_sims, num_frames, features_dataset, action_dataset):
    counter = 0
    for run in tqdm(iterators[mode]):
        actions = np.load(run + "/actions.npy")
        imgs = sorted(glob.glob('{}/*.jpg'.format(run)))
        assert len(actions) == len(imgs)-1

        for i in range(0, len(actions), num_frames)[:-1]:
            obs_np = []
            for j in range(i, i+num_frames):
                img = scipy.ndimage.imread(imgs[j])
                img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                obs_np.append(img)
            obs_np = np.stack(obs_np)  # (num_frames, 64, 64, 3)
            act_np = actions[i:i+num_frames-1]  # (num_frames, 5)

            features_dataset[:, counter, :, :, :] = obs_np  # (num_frames, 64, 64, 3)
            action_dataset[:, counter, :] = act_np  # (num_frames-1, 5)
            counter += 1