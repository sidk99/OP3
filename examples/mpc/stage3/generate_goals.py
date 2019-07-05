import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
import pickle
import torch
import os
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

def get_goal_info_seed_steps(env):
    frames = []
    actions = []

    env.create_tower_shape()
    goal_image = env.get_observation()
    goal_env_info = env.get_env_info()
    env.move_blocks_side()
    starting_info = env.get_env_info()
    frames.append(env.get_observation())

    num_seed_actions = 4
    for i in range(num_seed_actions):
        ac = env.sample_action("pick_block")
        frames.append(env.step(ac))
        actions.append(ac)

    frames = np.array(frames)
    actions = np.array(actions)

    return goal_image, frames, actions, goal_env_info, starting_info



def create_simple_goals():
    parser = ArgumentParser()
    parser.add_argument('-no', '--num_objects', type=int, required=True)
    parser.add_argument('-ng', '--num_goals', type=int, required=True)
    args = parser.parse_args()

    output_dir = get_module_path()

    env = BlockPickAndPlaceEnv(num_objects=args.num_objects, num_colors=None, img_dim=64, include_z=False,
                               random_initialize=False, view=False)

    # Creating dataset
    n_goals = args.num_goals
    env_data = []
    folder = output_dir + '/examples/mpc/stage3/goals/objects_seed_{}/'.format(args.num_objects)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(n_goals):
        goal_image, env_info = get_goal_info(env)
        misc.imsave(folder + 'img_{}.png'.format(i), goal_image)
        env_data.append(env_info)

    np.save(folder + 'env_data.npy', env_data)

    # Loading dataset example
    env_data = np.load(folder + 'env_data.npy')
    for i in range(min(10, n_goals)):
        env.set_env_info(env_data[i])  # Recreate the env with the correct blocks
        env.drop_heights = 3
        # for k in range(20): #The initial env has the full tower built, so we need to perturb it
        # initially
        # env.step(env.sample_action("pick_block"))
        optimal_actions = env.move_blocks_side()
        ob = env.get_observation()
        misc.imsave(folder + 'start_img_{}.png'.format(i), ob)

        for j in range(len(optimal_actions)):
            env.step(optimal_actions[j])
        ob = env.get_observation()
        misc.imsave(folder + 'rec_img_{}.png'.format(i), ob)

def create_seed_step_goals():
    parser = ArgumentParser()
    parser.add_argument('-no', '--num_objects', type=int, required=True)
    parser.add_argument('-ng', '--num_goals', type=int, required=True)
    args = parser.parse_args()

    output_dir = get_module_path()

    env = BlockPickAndPlaceEnv(num_objects=args.num_objects, num_colors=None, img_dim=64, include_z=False,
                               random_initialize=False, view=False)

    # Creating dataset
    n_goals = args.num_goals
    goal_dict = {"goal_image": [], "frames": [], "actions": [], "goal_env_info": [], "starting_env_info": []}
    folder = output_dir + '/examples/mpc/stage3/goals/objects_seed_{}/'.format(args.num_objects)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(n_goals):
        goal_image, frames, actions, goal_env_info, starting_env_info = get_goal_info_seed_steps(env)
        goal_dict["goal_image"].append(goal_image)
        goal_dict["frames"].append(frames)
        goal_dict["actions"].append(actions)
        goal_dict["goal_env_info"].append(goal_env_info)
        goal_dict["starting_env_info"].append(starting_env_info)

        misc.imsave(folder + 'goal_img_{}.png'.format(i), goal_image)
        misc.imsave(folder + 'initial_img_{}.png'.format(i), frames[0])
        misc.imsave(folder + 'after_seed_img_{}.png'.format(i), frames[-1])

    with open(folder+'goal_data.pkl', 'wb') as f:
        pickle.dump(goal_dict, f)


#python generate_goals.py -no 2 -ng 100
if __name__ == "__main__":
    create_seed_step_goals() #New version of goals with seed frames, actions, etc

    # create_simple_goals() #Old version of goals without seed frames, actions, etc


