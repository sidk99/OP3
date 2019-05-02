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

class MPC:

    def __init__(self, model, env, n_actions, mpc_steps):
        self.model = model
        self.env = env
        self.n_actions = n_actions
        self.mpc_steps = mpc_steps

    def run(self, goal_image):
        goal_image_tensor = ptu.from_numpy(np.moveaxis(goal_image, 2, 0)).unsqueeze(0).float() / 255. # (1, 3, imsize, imsize)
        rec_goal_image, goal_latents = self.model.refine(goal_image_tensor, hidden_state=None)  # (K, rep_size)

        obs = self.env.reset()

        obs_lst = [np.moveaxis(goal_image.astype(np.float32) / 255., 2, 0)]
        pred_obs_lst = [ptu.get_numpy(rec_goal_image)]


        for mpc_step in range(self.mpc_steps):
            pred_obs, action, goal_idx = self.step_mpc(obs, goal_latents, goal_image_tensor, mpc_step)
            obs = self.env.step(action)
            pred_obs_lst.append(pred_obs)
            obs_lst.append(np.moveaxis(obs, 2, 0))
            if goal_latents.shape[0] == 1:
                break
            # remove matching goal latent from goal latents
            goal_latents = torch.stack([goal_latents[i] for i in set(range(goal_latents.shape[0])) - set([goal_idx])])

        save_image(ptu.from_numpy(np.stack(obs_lst + pred_obs_lst)),
                    logger.get_snapshot_dir() + '/mpc.png', nrow=len(obs_lst))


    def cost_func(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)


    def best_action_latent(self, goal_latents, goal_image, pred_latents, pred_image, actions):
        # obs_latents is (n_actions, K, rep_size)
        # pred_obs is (n_actions, 3, imsize, imsize)
        best_goal_idx = 0
        best_action_idx = 0
        best_cost = np.inf
        best_latent_idx = 0

        # Compare against each goal latent
        for i in range(goal_latents.shape[0]):

            cost = self.cost_func(goal_latents[i].view(1, 1, -1), pred_latents) # cost is (n_actions, K)
            cost, latent_idx = cost.min(-1)  # take min among K

            min_cost, action_idx = cost.min(0)  # take min among n_actions

            if min_cost <= best_cost:
                best_goal_idx = i
                best_action_idx = action_idx
                best_cost = min_cost
                best_latent_idx = latent_idx[action_idx]

        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx])

        return best_pred_obs, actions[best_action_idx], best_goal_idx

    def best_action_pixel(self, goal_latents, goal_image, pred_latents, pred_image, actions):
        mse = torch.pow(pred_image - goal_image, 2).mean(3).mean(2).mean(1)

        min_cost, action_idx = mse.min(0)

        return ptu.get_numpy(pred_image[action_idx]), actions[action_idx], 0


    def model_step_batched(self, obs, actions, bs=8):
        # Handle large obs in batches
        n_batches = obs.shape[0] // bs
        outputs = [[], []]

        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, obs.shape[0])

            pred_obs, obs_latents = self.model.step(obs[start_idx:end_idx], actions[start_idx:end_idx])
            outputs[0].append(pred_obs)
            outputs[1].append(obs_latents)

        return torch.cat(outputs[0]).data, torch.cat(outputs[1]).data

    def step_mpc(self, obs, goal_latents, goal_image, mpc_step):
        # obs is (imsize, imsize, 3)
        # goal latents is (<K, rep_size)
        actions = np.stack([self.env.sample_action() for _ in range(self.n_actions)])

        true_actions = np.array([[1, 0, 0, -.5,  0, 0, 0, 0, 1, 0, .75, .75, 0],
                                 [1, 0, 0, -.75, 0, 1, 0, 0, 1, 0, .25, .75, .25],
                                 [1, 0, 0, .75,  0, 0, 0, 0, 1, 0, 0.5, .25, 1],
                                 [1, 0, 0, .75,  0, 1, 0, 0, 1, 0, 1.0, .25, .5]])
        #
        actions = np.concatenate([true_actions[mpc_step].reshape((1, -1)), actions])

        obs_rep = ptu.from_numpy(np.moveaxis(obs, 2, 0)).unsqueeze(0).repeat(actions.shape[0], 1, 1, 1)
        pred_obs, obs_latents = self.model_step_batched(obs_rep,
                                                   ptu.from_numpy(actions))


        save_image(pred_obs, logger.get_snapshot_dir() + '/mpc_pred_%d.png' %mpc_step)

        best_pred_obs, best_action, best_goal_idx = self.best_action_latent(goal_latents, goal_image,
                                                                     obs_latents, pred_obs, actions)

        return best_pred_obs, best_action, best_goal_idx


def main(variant):

    #model_file = variant['model_file']
    #goal_file = variant['goal_file']

    model_file = '/home/jcoreyes/objects/rlkit/output/04-25-iodine-blocks-physics-actions/04-25-iodine-blocks-physics-actions_2019_04_25_11_36_24_0000--s-98913/params.pkl'
    goal_file = '/home/jcoreyes/objects/object-oriented-prediction/o2p2/planning/executed/mjc_4.png'

    model = pickle.load(open(model_file, 'rb'))
    #model.cuda()

    env = BlockEnv(5)
    mpc = MPC(model, env, n_actions=15, mpc_steps=4)


    goal_image = imageio.imread(goal_file)
    mpc.run(goal_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--modelfile', type=str, default=None)
    parser.add_argument('-g', '--goalfile', type=str, default=None)
    args = parser.parse_args()

    variant = dict(
        algorithm='MPC',
        modelfile=args.modelfile,
        goalfile=args.goalfile,
    )


    run_experiment(
        main,
        exp_prefix='mpc',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



