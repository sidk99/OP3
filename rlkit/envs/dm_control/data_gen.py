from dm_control import suite, viewer
import numpy as np
from planet import control

import collections
import functools
import os
import argparse
import pathos.pools as pp
import h5py

Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = params.get('num_frames')
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup')
  return Task('cartpole_swingup', env_ctor, max_length, state_components)

def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 6)
  max_length = params.get('num_frames')
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch')
  return Task('cup_catch', env_ctor, max_length, state_components)

def _dm_control_env(action_repeat, max_length, domain, task):
  env = control.wrappers.DeepMindWrapper(suite.load(domain, task), (64, 64))
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.MaximumDuration(env, max_length)
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env



def createSingleSim(args):
    if args.env == 'cartpole_swingup':
        task = cartpole_swingup({}, dict(num_frames=args.num_frames))
    elif args.env == 'cup_catch':
        task = cartpole_swingup({}, dict(num_frames=args.num_frames))
    episodes = control.random_episodes(task.env_ctor, 1)

    return episodes[0]['image'], episodes[0]['action']



def createMultipleSims(args):
    datasets = {'training':args.num_sims,'validation':min(args.num_sims, 10)}
    image_res = args.img_dim
    pool = pp.ProcessPool(args.num_workers)
    with h5py.File(args.filename, 'w') as f:
        for folder in datasets:
            num_sims = datasets[folder]
            results = pool.map(createSingleSim, [args for _ in range(num_sims)])
            cur_folder = f.create_group(folder)
            # create datasets, write to disk
            image_data_shape = (args.num_frames, num_sims, image_res, image_res, 3)
            #groups_data_shape = (n_frames, num_sims, image_res, image_res, 1)
            action_data_shape = (args.num_frames, num_sims, results[0][1].shape[-1])
            features_dataset = cur_folder.create_dataset('features', image_data_shape, dtype='uint8')
            action_dataset = cur_folder.create_dataset('actions', action_data_shape, dtype='float32')


            for i in range(num_sims):
                frames, action_vec = results[i] #(T, M, N, C), (T, M, N, 1)
                # RV: frames is numpy with shape (T, M, N, C)
                # Bouncing balls dataset has shape (T, 1, M, N, C)
                #frames = np.expand_dims(frames, 1)

                features_dataset[:, i, :, :, :] = frames
                action_dataset[:, i, :] = action_vec

            print("Done with dataset: {}".format(folder))


parser = argparse.ArgumentParser()
## stuff you might want to edit
parser.add_argument('--num_frames', default=10, type=int,
        help='total number of frames per episode')
parser.add_argument('--num_sims', default=1, type=int,
        help='total number of episodes')
parser.add_argument('--img_dim', default=64, type=int,
        help='image dimension')
parser.add_argument('--env', default='cartpole_swingup', type=str)
parser.add_argument('--filename', default="dm_control.h5", type=str,   #RV: Added
        help='Name of the file. Please include .h5 at the end.')
parser.add_argument('-p', '--num_workers', type=int, default=1)
args = parser.parse_args()

createMultipleSims(args)