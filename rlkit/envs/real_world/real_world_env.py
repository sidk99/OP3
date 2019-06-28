import argparse
import cv2
import glob
import h5py
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

import dataset_utils as du

from poke import read_poke, write_poke
from kevin import read_kevin, write_kevin
from solid import read_solid, write_solid

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='kevin', help='path for hdf5 file')
    parser.add_argument('--num_frames', default=20, type=int,
            help='total number of frames per episode. Only applicable for Poke dataset')
    args = parser.parse_args()
    return args

def convert_data(args, reader, writer):
    fname = os.path.join(args.root, args.filename)
    print("Reading data")
    num_sims, hps, iterators, mode_map = reader(args)
    print(num_sims)
    print(hps.hp)
    print("Writing data")
    num_frames = hps.get('num_frames')
    image_res = hps.get('image_res')
    action_dim = hps.get('action_dim')
    datasets = {'training':num_sims[mode_map['training']],'validation':num_sims[mode_map['validation']]}
    with h5py.File(fname, 'w') as f:
        for folder in datasets:
            num_sims = datasets[folder]
            cur_folder = f.create_group(folder)
            image_data_shape = (num_frames, num_sims, image_res, image_res, 3)
            action_data_shape = (num_frames-1, num_sims, action_dim)
            features_dataset = cur_folder.create_dataset('features', image_data_shape, dtype='uint8')
            action_dataset = cur_folder.create_dataset('actions', action_data_shape, dtype='float32')

            mode = mode_map[folder]
            writer(iterators, mode, num_sims, num_frames, features_dataset, action_dataset)

    print('Finished writing to {}'.format(fname))

def make_gifs(fname, num_examples):
    dirname = os.path.dirname(fname)
    print("Done with creating the dataset, now creating visuals")
    du.hdf5_to_image(fname, num_examples)
    for i in tqdm(range(num_examples)):
        du.make_gif(os.path.join(dirname, "imgs/training/{}/features".format(str(i))), "animation.gif")

if __name__ == '__main__':
    args = parse_args()

    if args.root == 'kevin':
        """
        {'train': 1500, 'test': 0, 'val': 166}
        {'num_frames': 15, 'image_res': 64, 'action_dim': 2}
        """
        args.filename = 'kevin.h5'
        args.train_path = '5k_push_0.tfrecord'
        args.val_path = '5k_push_0_val.tfrecord'
        reader = read_kevin
        writer = write_kevin
    elif args.root == 'poke':
        """
        {'test': 795, 'train': 10872}
        {'num_frames': 20, 'action_dim': 5, 'image_res': 64}
        """
        args.filename = 'poke.h5'
        reader = read_poke
        writer = write_poke
    elif args.root == 'solid':
        """
        {'train': 7143, 'val': 414, 'test': 374}
        {'image_res': 64, 'action_dim': 4, 'num_frames': 30}
        """
        args.root = 'solid'
        args.filename = 'solid.h5'
        reader = read_solid
        writer = write_solid
    else:
        assert False

    convert_data(args, reader, writer)
    print("Done with creating the dataset, now creating visuals")
    make_gifs(os.path.join(args.root, args.filename), num_examples=20)

