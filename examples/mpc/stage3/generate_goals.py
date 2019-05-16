import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
import imageio
from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
from rlkit.core import logger
from torchvision.utils import save_image
from rlkit.util.plot import plot_multi_image
import scipy.misc as misc

# [6, 8, 11, 12, 17, 23, 29, 32, 34, 44, 50, 51, 54, 58, 61, 62, 67, 74, 83, 84, 96, 97, 105, 106, 117, 184, 187, 215,
# 219, 225, 234, 252, 258, 545, 537, 552, 613]
# goals_3 [12, 16]
# def rollout(env):
#     actions = []
#
#     env.create_tower_shape()
#     goal_image = env.get_observation()
#     env_info = env.get_env_info()
#
#
#     # obs = env.reset()
#     #n_obs = np.random.randint(3, 6)
#     for j in range(3):
#         action = env.sample_action()
#         obs = env.step(action)
#         actions.append(action)
#     return obs, np.stack(actions)


def get_goal_info(env):
    env.create_tower_shape()
    goal_image = env.get_observation()
    env_info = env.get_env_info()
    return goal_image, env_info


if __name__ == "__main__":

    # output_dir = '/home/jcoreyes/objects'
    output_dir = '/Users/aiflab/Desktop/Berkeley/Research/Rlkit'

    env = BlockPickAndPlaceEnv(num_objects=5, num_colors=5, img_dim=64, include_z=False, random_initialize=False, view=False)

    env_data = []
    for i in range(10):
        goal_image, env_info = get_goal_info(env)
        misc.imsave(output_dir + '/examples/mpc/stage3/goals/img_{}.png'.format(i), goal_image)
        env_data.append(env_info)

    np.save(output_dir + '/examples/mpc/stage3/goals/actions.npy', env_data)

    env_data = np.load(output_dir + '/examples/mpc/stage3/goals/actions.npy')
    for i in range(10):
        env.set_env_info(env_data[i])
        env.drop_heights = 3
        for k in range(20):
            env.step(env.sample_action("pick_block"))
        ob = env.get_observation()
        misc.imsave(output_dir + '/examples/mpc/stage3/goals/start_img_{}.png'.format(i), ob)

