import rlkit.torch.pytorch_util as ptu
from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
from torch.distributions import Normal
import pickle
import torch
import torch.nn as nn
from argparse import ArgumentParser
from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
import imageio
from rlkit.core import logger
from torchvision.utils import save_image
from rlkit.util.plot import plot_multi_image
import json
import os
import rlkit.torch.iodine.iodine as iodine
from collections import OrderedDict
from rlkit.util.misc import get_module_path
import pdb
import random



def main():
    return -1

def get_str_env_positions(env):
    the_str = ""
    for aname in env.names:
        the_str += "{}: {:.2f}, {:.2f}, {:.2f}  ".format(aname, *env.get_block_info(aname)["pos"])
    return the_str

def add_to_file(str_data, file_name):
    with open(file_name, "a") as myfile:
        myfile.write(str_data + "\n")


def single_exp_info(env_info, mpc_actions, log_file_name):
    env = BlockPickAndPlaceEnv(num_objects=1, num_colors=None, img_dim=64,
                               include_z=False)  # Note num_objects & num_colors do not matter
    env.set_env_info(env_info)  # Places the correct blocks in the environment, blocks will also be set in the goal position

    #Format of log files:
    # Goal block info
    # Starting block info
    # Action, blocks info

    add_to_file(get_str_env_positions(env), log_file_name) #Add goal info
    true_actions = env.move_blocks_side()  # Moves blocks to the side for mpc, returns true optimal actions
    add_to_file(get_str_env_positions(env), log_file_name)  # Add starting info

    for an_action in mpc_actions:
        add_to_file("Action: {}".format(an_action), log_file_name) #Add action
        [add_to_file("Moving: {}".format(aname), log_file_name) for aname in env.names if env.intersect(aname, an_action[:3])]
        env.step(an_action)
        add_to_file(get_str_env_positions(env), log_file_name) #Add result


def get_action_list(variant):
    module_path = get_module_path()
    result_path_name = '/nfs/kun1/users/rishiv/Research/op3_exps/06-19-mpc-stage3/06-19-mpc_stage3_2019_06_19_00_00_15_0000--s-31772/'
    action_list = np.load(result_path_name + 'optimal_actions.npy')

    for goal_idx in range(0, 1):
        env_info = np.load(module_path + '/examples/mpc/stage3/goals/env_data.npy')[goal_idx]
        single_exp_info(env_info, action_list[goal_idx], logger.get_snapshot_dir()+"/goalresults_{}.txt".format(goal_idx))




#CUDA_VISIBLE_DEVICES=7 python viz.py

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--modelfile', type=str, default=None)
    args = parser.parse_args()

    variant = dict(
    )

    run_experiment(
        get_action_list,
        exp_prefix='mpc_stage3_viz',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=None,
    )


