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


def get_goal_info(env):
    env.create_tower_shape()
    goal_image = env.get_observation()
    env_info = env.get_env_info()
    return goal_image, env_info


if __name__ == "__main__":

    output_dir = '/home/jcoreyes/objects/rlkit'
    #output_dir = '/Users/aiflab/Desktop/Berkeley/Research/Rlkit'

    env = BlockPickAndPlaceEnv(num_objects=4, num_colors=5, img_dim=64, include_z=False,
                               random_initialize=False, view=False)

    #Creating dataset
    env_data = []
    for i in range(10):
        goal_image, env_info = get_goal_info(env)
        misc.imsave(output_dir + '/examples/mpc/stage3/goals/img_{}.png'.format(i), goal_image)
        env_data.append(env_info)

    np.save(output_dir + '/examples/mpc/stage3/goals/actions.npy', env_data)

    # Loading dataset example
    env_data = np.load(output_dir + '/examples/mpc/stage3/goals/actions.npy')
    for i in range(10):
        env.set_env_info(env_data[i]) #Recreate the env with the correct blocks
        env.drop_heights = 3
        for k in range(20): #The initial env has the full tower built, so we need to perturb it initially
            env.step(env.sample_action("pick_block"))
        ob = env.get_observation()
        misc.imsave(output_dir + '/examples/mpc/stage3/goals/start_img_{}.png'.format(i), ob)

