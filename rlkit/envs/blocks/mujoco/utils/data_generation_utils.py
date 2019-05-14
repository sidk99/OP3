import os
import argparse
import pickle
import random
import tqdm
import pdb
import sys
import h5py
import shutil

import mujoco_py as mjc
import matplotlib.pyplot as plt

# sys.path.append("..")
import numpy as np
import imageio
import cv2

def createMultipleSims(args, obs_size, ac_size, createSingleSim):
    datasets = {'training':args.num_sims,'validation':min(args.num_sims, 100)}
    n_frames = args.num_frames
    with h5py.File(args.filename, 'w') as f:
        for folder in datasets:
            cur_folder = f.create_group(folder)

            num_sims = datasets[folder]

            # create datasets, write to disk
            # image_data_shape = (n_frames, num_sims, image_res, image_res, 3)
            image_data_shape = [n_frames, num_sims] + obs_size + [3]
            # groups_data_shape = [n_frames, num_sims] + list(env.get_obs_size()) + [1]
            # action_data_shape = (1, num_sims, len(polygons) + 7 + 3)
            action_data_shape = [n_frames, num_sims] + ac_size
            features_dataset = cur_folder.create_dataset('features', image_data_shape, dtype='float32')
            # groups_dataset = cur_folder.create_dataset('groups', groups_data_shape, dtype='float32')
            action_dataset = cur_folder.create_dataset('actions', action_data_shape, dtype='float32')

            # for i in range(num_sims):
            i = 0
            while i < num_sims:
                frames, action_vec = createSingleSim()  # (T, M, N, C), (T, A)
                # frames, group_frames, action_vec = createSingleSim() #(T, M, N, C), (T, M, N, 1)
                # pdb.set_trace()

                # RV: frames is numpy with shape (T, M, N, C)
                # Bouncing balls dataset has shape (T, 1, M, N, C)
                frames = np.expand_dims(frames, 1)
                # group_frames = np.expand_dims(group_frames, 1)

                features_dataset[:, [i], :, :, :] = frames
                # groups_dataset[:, [i], :, :, :] = group_frames
                action_dataset[:, i, :] = action_vec
                i += 1

            print("Done with dataset: {}".format(folder))


def make_gif(images_root, gifname):
    file_names = [fn for fn in os.listdir(images_root) if fn.endswith('.png')]
    file_names =  sorted(file_names, key=lambda x: int(os.path.splitext(x)[0]))
    # images = [Image.open(os.path.join(images_root,fn)) for fn in file_names]
    images = []
    for a_file in file_names:
        images.append(imageio.imread(os.path.join(images_root,a_file)))
    imageio.mimsave(images_root + gifname, images)

def mkdirp(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder

def hdf5_to_image(filename):
    root = os.path.dirname(filename)
    # shutil.rmtree(os.path.join(root, 'imgs'))
    img_root = mkdirp(os.path.join(root, 'imgs'))
    h5file = h5py.File(filename, 'r')
    for mode in h5file.keys():
        mode_folder = mkdirp(os.path.join(img_root, mode))
        groups = h5file[mode]
        f = groups['features']
        for ex in range(f.shape[1]):
            ex_folder = mkdirp(os.path.join(mode_folder, str(ex)))
            for d in groups.keys():
                if d in ['features', 'groups', 'collisions']:
                    dataset_folder = mkdirp(os.path.join(ex_folder, d))
                    dataset = groups[d]
                    num_groups = np.max(dataset[:, ex])
                    for j in range(dataset.shape[0]):
                        imfile = os.path.join(dataset_folder, str(j)+'.png')
                        if d == 'features':
                            cv2.imwrite(imfile, dataset[j, ex]*255)
                        elif d == 'groups':
                            cv2.imwrite(imfile, dataset[j, ex]*255.0/(num_groups))
                        elif d == 'collisions':
                            cv2.imwrite(imfile, dataset[j, ex]*255)
                        else:
                            assert False



# createMultipleSims()
# print("Done with creating the dataset, now creating visuals")
# hdf5_to_image(filename)
# for i in range(10):
#     tmp = os.path.join(args.output_path, "imgs/training/{}/features".format(str(i)))
#     make_gif(tmp, "animation.gif")
#     tmp = os.path.join(args.output_path, "imgs/training/{}/groups".format(str(i)))
#     make_gif(tmp, "animation.gif")


