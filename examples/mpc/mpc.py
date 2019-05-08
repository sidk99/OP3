import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
from torch.distributions import Normal
import pickle
import torch
from argparse import ArgumentParser
import imageio
from rlkit.envs.blocks.mujoco.block_stacking_env import BlockEnv
from rlkit.core import logger
from torchvision.utils import save_image
from rlkit.util.plot import plot_multi_image
import json
import os

class Cost:

    def __init__(self, type, logger_prefix_dir):
        self.type = type
        self.remove_goal_latents = True
        self.logger_prefix_dir = logger_prefix_dir

    def best_action(self, mpc_step, goal_latents, goal_latents_recon, goal_image, pred_latents,
             pred_latents_recon, pred_image, actions):
        if self.type == 'min_min_latent':
            return self.min_min_latent(goal_latents, goal_image, pred_latents, pred_latents_recon, actions)
        if self.type == 'sum_goal_min_latent':
            self.remove_goal_latents = False
            return self.sum_goal_min_latent(goal_latents, goal_image, pred_latents, pred_image, actions)

        elif self.type == 'goal_pixel':
            return self.goal_pixel(goal_latents, goal_image, pred_latents, pred_image, actions)
        elif self.type == 'latent_pixel':
            return self.latent_pixel(mpc_step, goal_latents_recon, goal_image, pred_latents_recon,
                                                 pred_image, actions)


    def mse(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

    def min_min_latent(self, goal_latents, goal_image, pred_latents, pred_image, actions):
        # obs_latents is (n_actions, K, rep_size)
        # pred_obs is (n_actions, 3, imsize, imsize)
        best_goal_idx = 0
        best_action_idx = 0
        best_cost = np.inf
        best_latent_idx = 0

        # Compare against each goal latent
        for i in range(goal_latents.shape[0]):

            cost = self.mse(goal_latents[i].view(1, 1, -1), pred_latents) # cost is (n_actions, K)
            cost, latent_idx = cost.min(-1)  # take min among K

            min_cost, action_idx = cost.min(0)  # take min among n_actions

            if min_cost <= best_cost:
                best_goal_idx = i
                best_action_idx = action_idx
                best_cost = min_cost
                best_latent_idx = latent_idx[action_idx]

        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx])

        return best_pred_obs, actions[best_action_idx], best_goal_idx

    def sum_goal_min_latent(self, goal_latents, goal_image, pred_latents, pred_image, actions):
        # obs_latents is (n_actions, K, rep_size)
        # pred_obs is (n_actions, 3, imsize, imsize)
        best_goal_idx = 0 # here this is meaningless

        # Compare against each goal latent
        costs = []
        for i in range(goal_latents.shape[0]):

            cost = self.mse(goal_latents[i].view(1, 1, -1), pred_latents) # cost is (n_actions, K)
            cost, latent_idx = cost.min(-1)  # take min among K
            costs.append(cost)

        _, best_action_idx = torch.stack(costs).sum(0).min(0)
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx])

        return best_pred_obs, actions[best_action_idx], best_goal_idx

    def goal_pixel(self, goal_latents, goal_image, pred_latents, pred_image, actions):
        mse = torch.pow(pred_image - goal_image, 2).mean(3).mean(2).mean(1)

        min_cost, action_idx = mse.min(0)

        return ptu.get_numpy(pred_image[action_idx]), actions[action_idx], 0

    def latent_pixel(self, mpc_step, goal_latents_recon, goal_image, pred_latents_recon, pred_image, actions):
        # obs_latents is (n_actions, K, rep_size)
        # pred_obs is (n_actions, 3, imsize, imsize)
        best_goal_idx = 0
        best_action_idx = 0
        best_cost = np.inf
        best_latent_idx = 0
        n_actions = actions.shape[0]
        K = pred_latents_recon.shape[1]

        imshape = (3, 64, 64)
        # Compare against each goal latent
        costs = []
        costs_latent = []
        latent_idxs = []
        for i in range(goal_latents_recon.shape[0]):
            cost = torch.pow(goal_latents_recon[i].view(1, 1, *imshape) - pred_latents_recon, 2).mean(4).mean(3).mean(2)
            costs_latent.append(cost)
            cost, latent_idx = cost.min(-1)  # take min among K
            costs.append(cost)
            latent_idxs.append(latent_idx)
            min_cost, action_idx = cost.min(0)  # take min among n_actions

            if min_cost <= best_cost:
                best_goal_idx = i
                best_action_idx = action_idx
                best_cost = min_cost
                best_latent_idx = latent_idx[action_idx]
        #import pdb; pdb.set_trace()
        costs = torch.stack(costs) # (n_goal_latents, n_actions )
        latent_idxs = torch.stack(latent_idxs) # (n_goal_latents, n_actions )

        matching_costs, matching_goal_idx = costs.min(0)
        #import pdb; pdb.set_trace()
        matching_latent_idx = latent_idxs[matching_goal_idx, np.arange(n_actions)]

        matching_goal_rec = torch.stack([goal_latents_recon[j] for j in matching_goal_idx])
        matching_latent_rec = torch.stack([pred_latents_recon[i][matching_latent_idx[i]] for i in range(n_actions)])
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx])

        full_plot = torch.cat([pred_image.unsqueeze(0),
                               pred_latents_recon.permute(1, 0, 2, 3, 4),
                               matching_latent_rec.unsqueeze(0),
                               matching_goal_rec.unsqueeze(0)], 0)

        plot_size = 8
        full_plot = full_plot[:, :plot_size]
        caption = np.zeros(full_plot.shape[:2])
        caption[1:1+K, :] = ptu.get_numpy(torch.stack(costs_latent).min(0)[0].permute(1, 0))[:, :plot_size]
        caption[-2, :] = matching_costs.cpu().numpy()[:plot_size]

        plot_multi_image(ptu.get_numpy(full_plot),
                         logger.get_snapshot_dir() + '%s/mpc_pred_%d.png' %(self.logger_prefix_dir, mpc_step), caption=caption)

        best_action_idxs = costs.min(0)[0].sort()[1]

        return best_pred_obs, actions[best_action_idxs], best_goal_idx

class MPC:

    def __init__(self, model, env, n_actions, mpc_steps,
                 n_goal_objs=3,
                 cost_type='latent_pixel',
                 filter_goals=False,
                 true_actions=None,
                 logger_prefix_dir=None,
                 mpc_style="random_shooting", #options are random_shooting, cem
                 cem_steps=2,
                 ):
        self.model = model
        self.env = env
        self.n_actions = n_actions
        self.mpc_steps = mpc_steps
        self.cost_type = cost_type
        self.filter_goals = filter_goals
        self.cost = Cost(self.cost_type, logger_prefix_dir)
        self.true_actions = true_actions
        self.n_goal_objs = n_goal_objs
        self.mpc_style = mpc_style
        self.cem_steps = cem_steps
        if logger_prefix_dir is not None:
            os.mkdir(logger.get_snapshot_dir() + logger_prefix_dir)
        self.logger_prefix_dir = logger_prefix_dir

    def filter_goal_latents(self, goal_latents, goal_latents_mask, goal_latents_recon):
        n_goals = self.n_goal_objs
        vals, keep = torch.sort(goal_latents_mask.mean(2).mean(1), descending=True)
        goal_latents_recon[keep[n_goals]] += goal_latents_recon[keep[n_goals+1]]
        keep = keep[1:1+n_goals]
        goal_latents = torch.stack([goal_latents[i] for i in keep])
        goal_latents_recon = torch.stack([goal_latents_recon[i] for i in keep])
        save_image(goal_latents_recon, logger.get_snapshot_dir() + '%s/mpc_goal_latents_recon.png' %self.logger_prefix_dir)

        return goal_latents, goal_latents_recon

    def remove_idx(self, array, idx):
        return torch.stack([array[i] for i in set(range(array.shape[0])) - set([idx])])

    def run(self, goal_image):
        goal_image_tensor = ptu.from_numpy(np.moveaxis(goal_image, 2, 0)).unsqueeze(0).float() / 255. # (1, 3, imsize, imsize)
        rec_goal_image, goal_latents, goal_latents_recon, goal_latents_mask = self.model.refine(goal_image_tensor, hidden_state=None,
                                                         plot_latents=False)  # (K, rep_size)

        # Keep top 4 goal latents with greatest mask area excluding 1st (background)
        #import pdb; pdb.set_trace()
        if self.filter_goals:
            goal_latents, goal_latents_recon = self.filter_goal_latents(goal_latents, goal_latents_mask, goal_latents_recon)

        obs = self.env.reset()

        obs_lst = [np.moveaxis(goal_image.astype(np.float32) / 255., 2, 0)]
        pred_obs_lst = [ptu.get_numpy(rec_goal_image)]

        actions = []
        for mpc_step in range(self.mpc_steps):
            pred_obs, action, goal_idx = self.step_mpc(obs, goal_latents, goal_image_tensor, mpc_step,
                                                       goal_latents_recon)
            actions.append(action)
            obs = self.env.step(action)
            pred_obs_lst.append(pred_obs)
            obs_lst.append(np.moveaxis(obs, 2, 0))
            if goal_latents.shape[0] == 1:
                break
            # remove matching goal latent from goal latents
            if self.cost.remove_goal_latents:
                goal_latents = self.remove_idx(goal_latents, goal_idx)
                goal_latents_recon = self.remove_idx(goal_latents_recon, goal_idx)

        save_image(ptu.from_numpy(np.stack(obs_lst + pred_obs_lst)),
                    logger.get_snapshot_dir() + '%s/mpc.png' %self.logger_prefix_dir, nrow=len(obs_lst))

        # Compare final obs to goal obs
        #import pdb; pdb.set_trace()
        mse = np.square(ptu.get_numpy(goal_image_tensor.squeeze().permute(1, 2, 0)) - obs).mean()

        return mse, np.stack(actions)


    def model_step_batched(self, obs, actions, bs=8):
        # Handle large obs in batches
        n_batches = int(np.ceil(obs.shape[0] / float(bs)))
        outputs = [[], [], []]

        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, obs.shape[0])

            pred_obs, obs_latents, obs_latents_recon = self.model.step(obs[start_idx:end_idx], actions[start_idx:end_idx])
            outputs[0].append(pred_obs)
            outputs[1].append(obs_latents)
            outputs[2].append(obs_latents_recon)

        return torch.cat(outputs[0]), torch.cat(outputs[1]), torch.cat(outputs[2])

    def step_mpc(self, obs, goal_latents, goal_image, mpc_step, goal_latents_recon):
        if self.mpc_style == 'random_shooting':
            best_pred_obs, best_actions, best_goal_idx = self._random_shooting_step(obs, goal_latents, goal_image,
                                                                                    mpc_step, goal_latents_recon)
            return best_pred_obs, best_actions[0], best_goal_idx
        elif self.mpc_style == 'cem':
            return self._cem_step(obs, goal_latents, goal_image, mpc_step, goal_latents_recon)

    def _cem_step(self, obs, goal_latents, goal_image, mpc_step, goal_latents_recon):

        actions = None
        filter_idx = int(self.n_actions * 0.1)
        for i in range(self.cem_steps):
            best_pred_obs, best_actions, best_goal_idx = self._random_shooting_step(obs, goal_latents, goal_image,
                                                                                    mpc_step, goal_latents_recon,
                                                                                    actions=actions)
            best_actions = best_actions[:filter_idx]
            mean = best_actions.mean(0)
            std = best_actions.std(0)
            actions = np.stack([self.env.sample_action_gaussian(mean, std) for _ in range(self.n_actions)])


        return best_pred_obs, best_actions[0], best_goal_idx



    def _random_shooting_step(self, obs, goal_latents, goal_image, mpc_step, goal_latents_recon,
                              actions=None):

        # obs is (imsize, imsize, 3)
        # goal latents is (<K, rep_size)
        if actions is None:
            actions = np.stack([self.env.sample_action() for _ in range(self.n_actions)])

        # polygox_idx, pos, axangle, rgb
        if self.true_actions is not None:
            actions = np.concatenate([self.true_actions[mpc_step].reshape((1, -1)), actions])

        obs_rep = ptu.from_numpy(np.moveaxis(obs, 2, 0)).unsqueeze(0).repeat(actions.shape[0], 1, 1, 1)
        pred_obs, obs_latents, obs_latents_recon = self.model_step_batched(obs_rep,
                                                   ptu.from_numpy(actions))

        best_pred_obs, best_actions, best_goal_idx = self.cost.best_action(mpc_step, goal_latents, goal_latents_recon, goal_image,
                                                                    obs_latents, obs_latents_recon, pred_obs, actions)

        return best_pred_obs, best_actions, best_goal_idx


def main(variant):

    #model_file = variant['model_file']
    #goal_file = variant['goal_file']

    model_file = '/home/jcoreyes/objects/rlkit/output/04-25-iodine-blocks-physics-actions/04-25-iodine-blocks-physics-actions_2019_04_25_11_36_24_0000--s-98913/params.pkl'
    #goal_file = '/home/jcoreyes/objects/object-oriented-prediction/o2p2/planning/executed/mjc_4.png'


    #goal_idxs = [i for i in range(20, 50)]
    goal_idxs = [26, 27, 28, 29, 30, 33, 51, 52, 55, 58, 59, 61, 62, 63, 65, 71, 81]



    model = pickle.load(open(model_file, 'rb'))
    #model.cuda()

    actions_lst = []
    stats = {'mse':0}
    for goal_idx in goal_idxs:
        goal_file = '/home/jcoreyes/objects/rlkit/examples/mpc/goals_3/img_%d.png' %goal_idx
        true_actions = np.load('/home/jcoreyes/objects/rlkit/examples/mpc/goals_3/actions.npy')[goal_idx]
        env = BlockEnv(5)
        mpc = MPC(model, env, n_actions=960, mpc_steps=3, true_actions=None,
              cost_type=variant['cost_type'], filter_goals=True, n_goal_objs=3, logger_prefix_dir='/goal_%d' % goal_idx,
                  mpc_style=variant['mpc_style'], cem_steps=5)
        goal_image = imageio.imread(goal_file)
        mse, actions = mpc.run(goal_image)
        stats['mse'] += mse
        actions_lst.append(actions)
        
    stats['mse'] /= len(goal_idxs)
    json.dump(stats, open(logger.get_snapshot_dir() + '/stats.json', 'w'))
    np.save(logger.get_snapshot_dir() + '/optimal_actions.npy', np.stack(actions_lst))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--modelfile', type=str, default=None)
    parser.add_argument('-g', '--goalfile', type=str, default=None)
    args = parser.parse_args()

    variant = dict(
        algorithm='MPC',
        modelfile=args.modelfile,
        goalfile=args.goalfile,
        cost_type='latent_pixel', # 'sum_goal_min_latent'
        mpc_style='cem',
    )


    run_experiment(
        main,
        exp_prefix='mpc',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



