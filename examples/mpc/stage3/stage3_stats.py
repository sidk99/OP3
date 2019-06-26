import rlkit.torch.pytorch_util as ptu
from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
from examples.mpc.stage3.mpc_stage3_v3 import Cost
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
from itertools import product
from torch.distributions import Normal
import pickle
import torch
import torch.nn as nn
from argparse import ArgumentParser
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


def create_grid(file_name):
    num_blocks = 1
    env = BlockPickAndPlaceEnv(num_objects=num_blocks, num_colors=None, img_dim=64, include_z=False, random_initialize=True)
    block_loc = env.get_block_info(env.names[0])['pos']

    bounds = env.bounds
    # self.bounds = {'x_min': -2.5, 'x_max': 2.5, 'y_min': 1.0, 'y_max': 4.0, 'z_min': 0.05, 'z_max':
    #     2.2}

    grid_points = 10

    x_vals = np.linspace(bounds['x_min'], bounds['x_max'], grid_points)
    y_vals = np.linspace(bounds['y_min'], bounds['y_max'], grid_points)


    # xx, yy = np.meshgrid(x_vals, y_vals)
    coords = product(x_vals, y_vals)

    actions = []
    obs = []
    for acoord in coords:
        action = np.array(list(block_loc) + list(acoord) + [env.drop_heights])
        # print(action)
        actions.append(action)
        obs.append(env.try_step(action)/255)

    obs = np.transpose(obs, (0, 3, 1, 2)) #(B, 3, D, D)

    results = {
        'actions': actions,
        'coords': coords,
        'obs': obs,
        'env_info': env.get_env_info(),
        'initial_obs': env.get_observation()/255
    }

    with open('resources/{}'.format(file_name), 'wb') as f:
        pickle.dump(results, f)
    return results

def analyze_grid(variant):
    with open('resources/{}'.format(variant['data_file_name']), 'rb') as f:
        results = pickle.load(f)
    obs = results['obs'] #(B, 3, D, D)
    actions = results['actions']

    m = load_model(variant)

    obs = ptu.from_numpy(obs).unsqueeze(1) #(B, T=1, 3, D, D)
    pred_obs, obs_latents, obs_latents_recon = m.step_batched(obs, None, bs=8)

    goal_image = obs[33:34]
    goal_ob, goal_latents, goal_latents_recon = m.step_batched(goal_image, None, bs=8)
    # pdb.set_trace()

    cost_class = Cost(type=variant['cost_type'], logger_prefix_dir="", latent_or_subimage='subimage', compare_func='mse',
                      post_process='negative_exp', aggregate='sum')
    sorted_costs, best_action_idxs, goal_latent_idxs = cost_class.get_action_rankings(goal_latents[0], goal_latents_recon[0], goal_image[0],
                                                                                      obs_latents, obs_latents_recon, pred_obs, image_suffix="analyze_grid",
                                                                                      plot_actions=8)


def load_model(variant):
    model_file = variant['model_file']
    module_path = get_module_path()

    m = iodine.create_model(variant, 4)
    state_dict = torch.load(model_file)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    m.load_state_dict(new_state_dict)
    m.cuda()
    m.set_eval_mode(True)

    return m

def run_analyze_grid():
    variant = dict(
        algorithm='MPC',
        model_file=args.modelfile,
        cost_type='sum_goal_min_latent_function',  # 'sum_goal_min_latent' 'latent_pixel 'sum_goal_min_latent_function'
        mpc_style='cem',  # random_shooting or cem
        model=iodine.imsize64_large_iodine_architecture,  # imsize64_large_iodine_architecture
        K=4,
        schedule_kwargs=dict(
            train_T=21,  # Number of steps in single training sequence, change with dataset
            test_T=21,  # Number of steps in single testing sequence, change with dataset
            seed_steps=4,  # Number of seed steps
            schedule_type='curriculum'  # single_step_physics, curriculum
        ),
        data_file_name='single_1.pkl'
    )

    run_experiment(
        analyze_grid,
        exp_prefix='mpc_stage3',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )


#CUDA_VISIBLE_DEVICES=7 python stage3_stats.py -f /nfs/kun1/users/rishiv/Research/op3_exps/06-10-iodine-blocks-pickplace-multienv-10k/06-10-iodine-blocks-pickplace_multienv_10k_2019_06_10_23_24_47_0000--s-18660/_params.pkl
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--modelfile', type=str, default=None)
    args = parser.parse_args()

    run_analyze_grid()
    # create_grid('single_1.pkl')

