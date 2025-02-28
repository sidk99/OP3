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
from rlkit.util.misc import get_module_path

def get_goal_info(env):
    env.create_tower_shape()
    goal_image = env.get_observation()
    env_info = env.get_env_info()
    return goal_image, env_info


if __name__ == "__main__":

    output_dir = get_module_path()

    env = BlockPickAndPlaceEnv(num_objects=3, num_colors=None, img_dim=64, include_z=False,
                               random_initialize=False, view=False)

    #Creating dataset
    n_goals = 2
    env_data = []
    for i in range(n_goals):
        goal_image, env_info = get_goal_info(env)
        misc.imsave(output_dir + '/examples/mpc/stage3/goals/img_{}.png'.format(i), goal_image)
        env_data.append(env_info)

    np.save(output_dir + '/examples/mpc/stage3/goals/actions.npy', env_data)

    # Loading dataset example
    env_data = np.load(output_dir + '/examples/mpc/stage3/goals/actions.npy')
    for i in range(n_goals):
        env.set_env_info(env_data[i]) #Recreate the env with the correct blocks
        env.drop_heights = 3
        #for k in range(20): #The initial env has the full tower built, so we need to perturb it
        # initially
            #env.step(env.sample_action("pick_block"))
        optimal_actions = env.move_blocks_side()
        ob = env.get_observation()
        misc.imsave(output_dir + '/examples/mpc/stage3/goals/start_img_{}.png'.format(i), ob)

        for j in range(len(optimal_actions)):
            env.step(optimal_actions[j])
        ob = env.get_observation()
        misc.imsave(output_dir + '/examples/mpc/stage3/goals/rec_img_{}.png'.format(i), ob)
