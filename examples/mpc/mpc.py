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


class Cost:

    def __init__(self, type):
        self.type = type
        self.remove_goal_latents = True

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
        caption = np.zeros(full_plot.shape[:2])
        caption[1:1+K, :] = ptu.get_numpy(torch.stack(costs_latent).min(0)[0].permute(1, 0))
        caption[-2, :] = matching_costs.cpu().numpy()

        plot_multi_image(ptu.get_numpy(full_plot),
                         logger.get_snapshot_dir() + '/mpc_pred_%d.png' % mpc_step, caption=caption)

        return best_pred_obs, actions[best_action_idx], best_goal_idx

class MPC:

    def __init__(self, model, env, n_actions, mpc_steps,
                 cost_type='latent_pixel',
                 filter_goals=False,
                 true_actions=None):
        self.model = model
        self.env = env
        self.n_actions = n_actions
        self.mpc_steps = mpc_steps
        self.cost_type = cost_type
        self.filter_goals = filter_goals
        self.cost = Cost(self.cost_type)
        self.true_actions = true_actions

    def filter_goal_latents(self, goal_latents, goal_latents_mask, goal_latents_recon,
                            n_goals=4):
        vals, keep = torch.sort(goal_latents_mask.mean(2).mean(1), descending=True)
        goal_latents_recon[keep[n_goals]] += goal_latents_recon[keep[n_goals+1]]
        keep = keep[1:1+n_goals]
        goal_latents = torch.stack([goal_latents[i] for i in keep])
        goal_latents_recon = torch.stack([goal_latents_recon[i] for i in keep])
        save_image(goal_latents_recon, logger.get_snapshot_dir() + '/mpc_goal_latents_recon.png')

        return goal_latents

    def remove_idx(self, array, idx):
        return torch.stack([array[i] for i in set(range(array.shape[0])) - set([idx])])

    def run(self, goal_image):
        goal_image_tensor = ptu.from_numpy(np.moveaxis(goal_image, 2, 0)).unsqueeze(0).float() / 255. # (1, 3, imsize, imsize)
        rec_goal_image, goal_latents, goal_latents_recon, goal_latents_mask = self.model.refine(goal_image_tensor, hidden_state=None,
                                                         plot_latents=True)  # (K, rep_size)

        # Keep top 4 goal latents with greatest mask area excluding 1st (background)
        #import pdb; pdb.set_trace()
        if self.filter_goals:
            goal_latents = self.filter_goal_latents(goal_latents, goal_latents_mask, goal_latents_recon)

        obs = self.env.reset()

        obs_lst = [np.moveaxis(goal_image.astype(np.float32) / 255., 2, 0)]
        pred_obs_lst = [ptu.get_numpy(rec_goal_image)]


        for mpc_step in range(self.mpc_steps):
            pred_obs, action, goal_idx = self.step_mpc(obs, goal_latents, goal_image_tensor, mpc_step,
                                                       goal_latents_recon)
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
                    logger.get_snapshot_dir() + '/mpc.png', nrow=len(obs_lst))




    def model_step_batched(self, obs, actions, bs=8):
        # Handle large obs in batches
        n_batches = obs.shape[0] // bs
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
        # obs is (imsize, imsize, 3)
        # goal latents is (<K, rep_size)
        actions = np.stack([self.env.sample_action() for _ in range(self.n_actions)])

        # polygox_idx, pos, axangle, rgb
        if self.true_actions is not None:
            actions = np.concatenate([self.true_actions[mpc_step].reshape((1, -1)), actions])

        obs_rep = ptu.from_numpy(np.moveaxis(obs, 2, 0)).unsqueeze(0).repeat(actions.shape[0], 1, 1, 1)
        pred_obs, obs_latents, obs_latents_recon = self.model_step_batched(obs_rep,
                                                   ptu.from_numpy(actions))

        best_pred_obs, best_action, best_goal_idx = self.cost.best_action(mpc_step, goal_latents, goal_latents_recon, goal_image,
                                                                    obs_latents, obs_latents_recon, pred_obs, actions)

        return best_pred_obs, best_action, best_goal_idx


def main(variant):

    #model_file = variant['model_file']
    #goal_file = variant['goal_file']

    model_file = '/home/jcoreyes/objects/rlkit/output/04-25-iodine-blocks-physics-actions/04-25-iodine-blocks-physics-actions_2019_04_25_11_36_24_0000--s-98913/params.pkl'
    goal_file = '/home/jcoreyes/objects/object-oriented-prediction/o2p2/planning/executed/mjc_4.png'

    model = pickle.load(open(model_file, 'rb'))
    #model.cuda()
    true_actions = np.array([
            [1, 0, 0, -.6, 0, 0, 0, 0, 1, 0, .75, .75, 0.5],
            [1, 0, 0, -.6, 0, 1, 0, 0, 1, 0, .25, .75, 1.0],
            [1, 0, 0, .6, 0,  0, 0, 0, 1, 0, 0.5, .5, 1],
            [1, 0, 0, .6, 0,  1, 0, 0, 1, 0, 0., .75, 0.75],

        ])
    env = BlockEnv(5)
    mpc = MPC(model, env, n_actions=7, mpc_steps=4, true_actions=true_actions,
              cost_type=variant['cost_type'])


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
        cost_type='sum_goal_min_latent'
    )


    run_experiment(
        main,
        exp_prefix='mpc',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



