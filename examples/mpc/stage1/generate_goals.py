import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
import imageio
from rlkit.envs.blocks.mujoco.block_stacking_env import BlockEnv
from rlkit.core import logger
from torchvision.utils import save_image
from rlkit.util.plot import plot_multi_image
import scipy.misc as misc

# [6, 8, 11, 12, 17, 23, 29, 32, 34, 44, 50, 51, 54, 58, 61, 62, 67, 74, 83, 84, 96, 97, 105, 106, 117, 184, 187, 215,
# 219, 225, 234, 252, 258, 545, 537, 552,, 613]
# goals_3 [12, 16]
def rollout(env):
    actions = []
    obs = env.reset()
    #n_obs = np.random.randint(3, 6)
    for j in range(3):
        action = env.sample_action()
        obs = env.step(action)
        actions.append(action)
    return obs, np.stack(actions)

if __name__ == "__main__":

    output_dir = '/home/jcoreyes/objects'
    env = BlockEnv(5)
    rollout_output = [[], []]
    for i in range(100):
        obs, actions = rollout(env)
        rollout_output[0].append(obs)
        rollout_output[1].append(actions)
        misc.imsave(output_dir + '/rlkit/examples/mpc/stage1/goals_3/img_%d.png' %i, obs)
    all_actions = np.stack(rollout_output[1])

    np.save(output_dir + '/rlkit/examples/mpc/stage1/goals_3/actions.npy', all_actions)

