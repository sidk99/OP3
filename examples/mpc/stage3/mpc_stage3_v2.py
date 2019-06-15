import rlkit.torch.pytorch_util as ptu
from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
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

class Cost:
    def __init__(self, type, logger_prefix_dir):
        self.type = type
        self.remove_goal_latents = False
        self.logger_prefix_dir = logger_prefix_dir

    def best_action(self, mpc_step, goal_latents, goal_latents_recon, goal_image, pred_latents,
                    pred_latents_recon, pred_image, actions, cem_step):
        self.cem_step = cem_step
        if self.type == 'min_min_latent':
            return self.min_min_latent(mpc_step, goal_latents, goal_latents_recon, goal_image,
                                            pred_latents, pred_latents_recon, pred_image, actions)
        elif self.type == 'sum_goal_min_latent':
            self.remove_goal_latents = False
            return self.sum_goal_min_latent(mpc_step, goal_latents, goal_latents_recon, goal_image,
                                            pred_latents, pred_latents_recon, pred_image, actions)
        elif self.type == 'sum_goal_min_latent_function':
            self.remove_goal_latents = False
            return self.sum_goal_min_latent_function(mpc_step, goal_latents, goal_latents_recon, goal_image,
                                            pred_latents, pred_latents_recon, pred_image, actions)
        elif self.type == 'goal_pixel':
            return self.goal_pixel(goal_latents, goal_image, pred_latents, pred_image, actions)
        elif self.type == 'latent_pixel':
            return self.latent_pixel(mpc_step, goal_latents_recon, goal_image, pred_latents_recon,
                                     pred_image, actions)
        else:
            raise Exception

    def mse(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

    def image_mse(self, im1, im2):
        #im1, im2 are (*, 3, D, D)
        return torch.pow(im1 - im2, 2).mean((-1, -2, -3)) #Takes means across the last

    def min_min_latent(self, mpc_step, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_image, actions):
        # obs_latents is (n_actions, K, rep_size)
        # pred_obs is (n_actions, 3, imsize, imsize)
        best_goal_idx = 0
        best_action_idx = 0
        best_cost = np.inf
        best_latent_idx = 0
        n_actions = actions.shape[0]
        K = pred_latents_recon.shape[1]

        rep_size = 128
        # Compare against each goal latent
        costs = []
        costs_latent = []
        latent_idxs = []
        for i in range(goal_latents.shape[0]):
            cost = torch.pow(goal_latents[i].view(1, 1, rep_size) - pred_latents,
                             2).mean(2)
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

        costs = torch.stack(costs)  # (n_goal_latents, n_actions )
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions )

        matching_costs, matching_goal_idx = costs.min(0)

        matching_latent_idx = latent_idxs[matching_goal_idx, np.arange(n_actions)]

        matching_goal_rec = torch.stack([goal_latents_recon[j] for j in matching_goal_idx])
        matching_latent_rec = torch.stack(
            [pred_latents_recon[i][matching_latent_idx[i]] for i in range(n_actions)])
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx])

        full_plot = torch.cat([pred_image.unsqueeze(0),
                               pred_latents_recon.permute(1, 0, 2, 3, 4),
                               matching_latent_rec.unsqueeze(0),
                               matching_goal_rec.unsqueeze(0)], 0)
        sort_costs, best_action_idxs = costs.min(0)[0].sort()
        plot_size = 8
        full_plot = full_plot[:, ptu.get_numpy(best_action_idxs[:plot_size])]
        caption = np.zeros(full_plot.shape[:2])
        caption[1:1 + K, :] = ptu.get_numpy(torch.stack(costs_latent).min(0)[0].permute(1, 0))[:,
                              :plot_size]
        caption[-2, :] = matching_costs.cpu().numpy()[ptu.get_numpy(best_action_idxs[:plot_size])]

        plot_multi_image(ptu.get_numpy(full_plot),
                         logger.get_snapshot_dir() + '%s/mpc_pred_%d.png' % (
                         self.logger_prefix_dir, mpc_step),
                         caption=caption)

        return best_pred_obs, actions[ptu.get_numpy(best_action_idxs)], best_goal_idx, sort_costs, best_action_idxs

    def sum_goal_min_latent(self,  mpc_step, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_image, actions):
        # obs_latents is (n_actions, K, rep_size)
        # pred_obs is (n_actions, 3, imsize, imsize)
        best_goal_idx = 0  # here this is meaningless
        n_actions = actions.shape[0]
        K = pred_latents_recon.shape[1]
        # Compare against each goal latent
        costs = []
        costs_latent = []
        latent_idxs = []
        for i in range(goal_latents.shape[0]):
            cost = self.mse(goal_latents[i].view(1, 1, -1), pred_latents)  # cost is (n_actions, K)
            costs_latent.append(cost)
            cost, latent_idx = cost.min(-1)  # take min among K
            costs.append(cost)
            latent_idxs.append(latent_idx)

        costs = torch.stack(costs)
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions )

        sorted_costs, best_action_idxs = costs.sum(0).sort()
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idxs[0]])

        matching_costs, matching_goal_idx = costs.min(0)

        matching_latent_idx = latent_idxs[matching_goal_idx, np.arange(n_actions)]

        matching_goal_rec = torch.stack([goal_latents_recon[j] for j in matching_goal_idx])
        matching_latent_rec = torch.stack(
            [pred_latents_recon[i][matching_latent_idx[i]] for i in range(n_actions)])

        full_plot = torch.cat([pred_image.unsqueeze(0),
                               pred_latents_recon.permute(1, 0, 2, 3, 4),
                               matching_latent_rec.unsqueeze(0),
                               matching_goal_rec.unsqueeze(0)], 0)

        plot_size = 8
        full_plot = full_plot[:, :plot_size]
        caption = np.zeros(full_plot.shape[:2])
        caption[1:1 + K, :] = ptu.get_numpy(torch.stack(costs_latent).min(0)[0].permute(1, 0))[:,:plot_size]
        caption[-2, :] = matching_costs.cpu().numpy()[:plot_size]

        plot_multi_image(ptu.get_numpy(full_plot),
                         logger.get_snapshot_dir() + '%s/mpc_pred_%d.png' % (
                             self.logger_prefix_dir, mpc_step),
                         caption=caption)

        return best_pred_obs, actions[ptu.get_numpy(best_action_idxs)], best_goal_idx, sorted_costs, best_action_idxs

    def sum_goal_min_latent_function(self,  mpc_step, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_image, actions, diff_type="subimage", function_type="raw"):
        # mpc_step: int
        # goal_latents: (n_goal_latents=K, rep_size)
        # goal_latents_recon: (n_goal_latents=K, 3, 64, 64)
        # goal_image: (1, 3, 64, 64)
        # pred_latents: (n_actions, K, rep_size)
        # pred_latents_recon: (n_actions, K, 3, 64, 64)
        # pred_image: (n_actions, 3, 64, 64)
        # actions: (n_actions, Th, A)

        best_goal_idx = 0  # here this is meaningless
        n_actions = actions.shape[0]
        K = pred_latents_recon.shape[1]
        # Compare against each goal latent
        costs = []  #(K, n_actions)
        distances = [] # (n_goal_latents=K, n_actions, K)
        latent_idxs = [] # (n_goal_latents=K, n_actions), [a,b] is an index corresponding to a latent
        for i in range(goal_latents.shape[0]): #Going through all n_goal_latents goal latents
            if diff_type == "subimage":
                dists = self.image_mse(goal_latents_recon[i].view((1, 1, 3, 64, 64)), pred_latents_recon) #(n_actions, K)
            elif diff_type == "latent":
                dists = self.mse(goal_latents[i].view(1, 1, -1), pred_latents)  # (n_actions, K)
            else:
                raise KeyError("Incorrect diff_type to sum_goal_min_latent_function: {}".format(diff_type))

            distances.append(dists)
            min_dist, latent_idx = dists.min(-1)  # take min among K, size is (n_actions)

            if function_type == "negative_exp":
                costs.append(-torch.exp(-min_dist))
            elif function_type == "raw":
                costs.append(min_dist)
            else:
                raise KeyError("Incorrect function_type to sum_goal_min_latent_function: {}".format(function_type))
            latent_idxs.append(latent_idx)

        # pdb.set_trace()
        costs = torch.stack(costs) # K, n_actions
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents=K, n_actions)
        distances = torch.stack(distances)
        #
        # best_action_idxs = costs.sum(0).sort()[1] #n_actions
        # best_pred_obs = ptu.get_numpy(pred_image[best_action_idxs[0]]) #(3, D, D)
        #
        # matching_costs, matching_goal_idx = costs.min(0) #Takes min across K -> both are (n_actions)
        # matching_latent_idx = latent_idxs[matching_goal_idx, np.arange(n_actions)]
        #
        # matching_goal_rec = torch.stack([goal_latents_recon[j] for j in matching_goal_idx])
        # matching_latent_rec = torch.stack([pred_latents_recon[i][matching_latent_idx[i]] for i in range(n_actions)])
        #
        # full_plot = torch.cat([pred_image.unsqueeze(0),  # (1, n_actions, 3, 64, 64)
        #                        pred_latents_recon.permute(1, 0, 2, 3, 4),  # (K, n_actions, 3, 64, 64)
        #                        matching_latent_rec.unsqueeze(0), #
        #                        matching_goal_rec.unsqueeze(0)], 0)

        #Sort by sum cost
        #Image contains the following: Pred_images, goal_latent_reconstructions, and
        # corresponding pred_latent_reconstructions
        #For every latent in goal latents, find corresponding predicted one (this is in latent_idxs)
        #  Should have something that is (K, num_actions) -> x[a,b] is index for pred_latents_recon
        #   pred
        sorted_costs, best_action_idxs = costs.sum(0).sort()
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idxs[0]])  # (3, D, D)
        sorted_pred_images = pred_image[best_action_idxs]
        # sorted_pred_latent_recons = pred_latents_recon[best_action_idxs]
        #Now we want sorted_pred_latent_recons[:, 0] to correspond to goal_latents_recon[0]
        corresponding_pred_latent_recons = []
        # corresponding_costs = []
        for i in range(goal_latents.shape[0]):
            tmp = pred_latents_recon[best_action_idxs, latent_idxs[i, best_action_idxs]] #(n_actions, 3, 64, 64)
            corresponding_pred_latent_recons.append(tmp)
            # corresponding_costs.append(costs[i, best_action_idxs])  # n_actions
            # corresponding_costs.append(costs[i, latent_idxs[i, :]]) #n_actions
            # pdb.set_trace()
        corresponding_pred_latent_recons = torch.stack(corresponding_pred_latent_recons) #(n_goal_latents=K, n_actions, 3, 64, 64)
        # corresponding_costs = torch.stack(corresponding_costs)
        # corresponding_costs = corresponding_costs[:, best_action_idxs]
        corresponding_costs = costs[:, best_action_idxs]
        # pdb.set_trace()
        full_plot = torch.cat([sorted_pred_images.unsqueeze(0), # (1, n_actions, 3, 64, 64)
                               corresponding_pred_latent_recons, # (n_goal_latents=K, n_actions, 3, 64, 64)
                               ], 0)
        # pdb.set_trace()
        plot_size = 8
        full_plot = full_plot[:, :plot_size]
        caption = np.zeros(full_plot.shape[:2])
        caption[0, :] = ptu.get_numpy(sorted_costs[:plot_size])
        caption[1:1+K, :] = ptu.get_numpy(corresponding_costs[:plot_size])[:,:plot_size]

        plot_multi_image(ptu.get_numpy(full_plot),
                         logger.get_snapshot_dir() + '{}/mpc_pred_{}_{}.png'.format(self.logger_prefix_dir, mpc_step,
                                                                                    self.cem_step), caption=caption)

        # pred_latents_recon = pred_latents_recon[best_action_idxs]
        # sorted_latents = pred_latents_recon[latent_idxs[0]]


        # plot_size = 8
        # full_plot = full_plot[:, :plot_size]
        # caption = np.zeros(full_plot.shape[:2])
        # # pdb.set_trace()
        # caption[0, :] = ptu.get_numpy(costs.sum(0).sort()[0][:plot_size])
        # caption[1:1 + K, :] = ptu.get_numpy(distances.min(0)[0].permute(1, 0))[:,:plot_size]
        # caption[-2, :] = ptu.get_numpy(matching_costs)[:plot_size]
        #
        # plot_multi_image(ptu.get_numpy(full_plot),
        #                  logger.get_snapshot_dir() + '{}/mpc_pred_{}_{}.png'.format(self.logger_prefix_dir, mpc_step,
        #                                                                             self.cem_step), caption=caption)

        return best_pred_obs, actions[ptu.get_numpy(best_action_idxs)], best_goal_idx, sorted_costs, best_action_idxs

    def goal_pixel(self, goal_latents, goal_image, pred_latents, pred_image, actions):
        mse = torch.pow(pred_image - goal_image, 2).mean(3).mean(2).mean(1)

        sorted_costs, action_idx = mse.min(0)

        return ptu.get_numpy(pred_image[action_idx]), actions[action_idx], 0, sorted_costs, action_idx

    def latent_pixel(self, mpc_step, goal_latents_recon, goal_image, pred_latents_recon, pred_image,
                     actions):
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
            cost = torch.pow(goal_latents_recon[i].view(1, 1, *imshape) - pred_latents_recon,
                             2).mean(4).mean(3).mean(2)
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

        costs = torch.stack(costs)  # (n_goal_latents, n_actions )
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions )
        sorted_costs, best_action_idxs = costs.min(0)[0].sort()

        matching_costs, matching_goal_idx = costs.min(0)

        matching_latent_idx = latent_idxs[matching_goal_idx, np.arange(n_actions)]

        matching_goal_rec = torch.stack([goal_latents_recon[j] for j in matching_goal_idx])
        matching_latent_rec = torch.stack(
            [pred_latents_recon[i][matching_latent_idx[i]] for i in range(n_actions)])
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx])

        full_plot = torch.cat([pred_image.unsqueeze(0),
                               pred_latents_recon.permute(1, 0, 2, 3, 4),
                               matching_latent_rec.unsqueeze(0),
                               matching_goal_rec.unsqueeze(0)], 0)

        plot_size = 8
        full_plot = full_plot[:, ptu.get_numpy(best_action_idxs[:plot_size])]
        caption = np.zeros(full_plot.shape[:2])
        caption[1:1 + K, :] = ptu.get_numpy(torch.stack(costs_latent).min(0)[0].permute(1, 0))[:,
                              :plot_size]
        caption[-2, :] = matching_costs.cpu().numpy()[ptu.get_numpy(best_action_idxs[:plot_size])]

        plot_multi_image(ptu.get_numpy(full_plot),
                         logger.get_snapshot_dir() + '{}/mpc_pred_{}_{}.png'.format(self.logger_prefix_dir, mpc_step, self.cem_step),
                         caption=caption)

        return best_pred_obs, actions[ptu.get_numpy(best_action_idxs)], best_goal_idx, sorted_costs, best_action_idxs


class MPC:
    def __init__(self, model, env, n_actions, mpc_steps,
                 n_goal_objs=3,
                 cost_type='latent_pixel',
                 filter_goals=False,
                 true_actions=None,
                 logger_prefix_dir=None,
                 mpc_style="random_shooting",  # options are random_shooting, cem
                 cem_steps=2,
                 use_action_image=True, # True for stage 1, False for stage 3
                 time_horizon=1 #How many steps into the future to do per step
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
        self.use_action_image = use_action_image
        self.time_horizon = time_horizon

    def filter_goal_latents(self, goal_latents, goal_latents_mask, goal_latents_recon):
        # Keep top goal latents with highest mask area except first
        n_goals = self.n_goal_objs
        goal_latents_mask[goal_latents_mask < 0.5] = 0
        vals, keep = torch.sort(goal_latents_mask.mean(2).mean(1), descending=True)

        goal_latents_recon[keep[n_goals]] += goal_latents_recon[keep[n_goals + 1]]
        keep = keep[1:1 + n_goals]
        goal_latents = torch.stack([goal_latents[i] for i in keep])
        goal_latents_recon = torch.stack([goal_latents_recon[i] for i in keep])

        save_image(goal_latents_recon,
                   logger.get_snapshot_dir() + '%s/mpc_goal_latents_recon.png' %
                   self.logger_prefix_dir)

        return goal_latents, goal_latents_recon

    def remove_idx(self, array, idx):
        return torch.stack([array[i] for i in set(range(array.shape[0])) - set([idx])])

    def run(self, goal_image):
        goal_image_tensor = ptu.from_numpy(np.moveaxis(goal_image, 2, 0)).unsqueeze(0).float()[:, :3] / 255.  # (1, 3, imsize, imsize)

        rec_goal_image, goal_latents, goal_latents_recon, goal_latents_mask = self.model.refine(
            goal_image_tensor,
            hidden_state=None,
            plot_latents=False)  # (K, rep_size)

        # Keep top 4 goal latents with greatest mask area excluding 1st (background)
        if self.filter_goals:
            goal_latents, goal_latents_recon = self.filter_goal_latents(goal_latents, goal_latents_mask, goal_latents_recon)

        #true_actions = self.env.move_blocks_side()
        #self.true_actions = true_actions
        obs = self.env.get_observation()
        import matplotlib.pyplot as plt
        # for i in range(4):
        #     action = true_actions[i]
        #
        #     print(action)
        #     obs = self.env.step(action)
        #     imageio.imsave(logger.get_snapshot_dir() + '%s/action_%d.png' %
        #                    (self.logger_prefix_dir, i), obs)
            #import pdb; pdb.set_trace()
        #true_actions[:, 2] = 0.2
        #true_actions[:, -1] = 3.5

        imageio.imsave(logger.get_snapshot_dir() + '{}/initial_image.png'.format(self.logger_prefix_dir), obs)
        obs_lst = [np.moveaxis(goal_image.astype(np.float32) / 255., 2, 0)[:3]]
        pred_obs_lst = [ptu.get_numpy(rec_goal_image)]

        chosen_actions = []
        for mpc_step in range(self.mpc_steps):
            pred_obs, actions, goal_idx = self.step_mpc(obs, goal_latents, goal_image_tensor, mpc_step, goal_latents_recon)
            # pdb.set_trace()
            chosen_actions.append(actions[0])
            for i in range(self.time_horizon):
                obs = self.env.step(actions[i])/255
            pred_obs_lst.append(pred_obs)
            obs_lst.append(np.moveaxis(obs, 2, 0))
            if goal_latents.shape[0] == 1:
                break
            # remove matching goal latent from goal latents
            if self.cost.remove_goal_latents:
                goal_latents = self.remove_idx(goal_latents, goal_idx)
                goal_latents_recon = self.remove_idx(goal_latents_recon, goal_idx)

        save_image(ptu.from_numpy(np.stack(obs_lst + pred_obs_lst)),
                   logger.get_snapshot_dir() + '%s/mpc.png' % self.logger_prefix_dir, nrow=len(obs_lst))

        # Compare final obs to goal obs
        mse = np.square(ptu.get_numpy(goal_image_tensor.squeeze().permute(1, 2, 0)) - obs).mean()

        return mse, np.stack(chosen_actions)

    #actions: n_actions, T, A
    #obs: n_actions, 3, D, D
    def model_step_batched(self, obs, actions, bs=4):
        # Handle large obs in batches
        n_batches = int(np.ceil(obs.shape[0] / float(bs)))
        outputs = [[], [], []]

        # old_env_info = self.env.get_env_info()
        # real_obs = []
        # for action in ptu.get_numpy(actions):
        #     self.env.set_env_info(old_env_info)
        #     actual_obs = self.env.step(action)
        #     real_obs.append(actual_obs)
        # self.env.set_env_info(old_env_info)


        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, obs.shape[0])
            actions_batch = actions[start_idx:end_idx] if not self.use_action_image else None

            pred_obs, obs_latents, obs_latents_recon = self.model.step(obs[start_idx:end_idx]/255., actions_batch, plot_latents=True)
            outputs[0].append(pred_obs)
            outputs[1].append(obs_latents)
            outputs[2].append(obs_latents_recon)

        # save_image(ptu.from_numpy(np.concatenate([np.moveaxis(np.stack(real_obs)/255., 3, 1), ptu.get_numpy(pred_obs)])),
        #            logger.get_snapshot_dir() + '{}/pred_vs_real.png'.format(self.logger_prefix_dir), nrow=7)

        return torch.cat(outputs[0]), torch.cat(outputs[1]), torch.cat(outputs[2])

    def step_mpc(self, obs, goal_latents, goal_image, mpc_step, goal_latents_recon):
        if self.mpc_style == 'random_shooting':
            best_pred_obs, best_actions, best_goal_idx = self._random_shooting_step(obs,
                                                                                    goal_latents,
                                                                                    goal_image,
                                                                                    mpc_step,
                                                                                    0,
                                                                                    goal_latents_recon)
            return best_pred_obs, best_actions[0], best_goal_idx
        elif self.mpc_style == 'cem':
            return self._cem_step(obs, goal_latents, goal_image, mpc_step, goal_latents_recon)

    def _cem_step(self, obs, goal_latents, goal_image, mpc_step, goal_latents_recon):

        actions = None
        filter_idx = int(self.n_actions * 0.1)
        print("self.cem_steps: {}".format(self.cem_steps))
        for i in range(self.cem_steps):
            best_pred_obs, best_actions, best_goal_idx = self._random_shooting_step(obs,
                                                                                    goal_latents,
                                                                                    goal_image,
                                                                                    mpc_step,
                                                                                    i,
                                                                                    goal_latents_recon,
                                                                                    actions=actions)
            # pdb.set_trace()
            best_actions = best_actions[:filter_idx] #(n_actions, Th, A)
            mean = best_actions.mean(0) #(Th, A)
            std = best_actions.std(0) #(Th, A)
            # actions = np.stack([self.env.sample_action_gaussian(mean, std) for _ in range(self.n_actions)])
            actions = self.env.sample_multiple_action_gaussian(mean, std, self.n_actions)

        return best_pred_obs, best_actions[0], best_goal_idx

    def _random_shooting_step(self, obs, goal_latents, goal_image, mpc_step, cem_step, goal_latents_recon,
                              actions=None):

        # obs is (imsize, imsize, 3)
        # goal latents is (<K, rep_size)
        if actions is None:
            actions = []
            for i in range(self.time_horizon):
                actions.append(np.stack([self.env.sample_action(action_type='pick_block') for _ in range(self.n_actions)]))
            actions = np.stack(actions) #T, num_actions, A
            actions = np.moveaxis(actions, (0, 1, 2), (1, 0, 2)) #num_actions, Th, A

        # print(actions.shape)
        # polygox_idx, pos, axangle, rgb
        if self.true_actions is not None:
            true_horizon_actions = [] #Will have size (Th, A)
            for i in range(mpc_step, mpc_step+self.time_horizon):
                if i < len(self.true_actions): #Add true best actions
                    true_horizon_actions.append(self.true_actions[i])
                else: #Add dummy best actions, note this could mess up the scene and actually be unoptimal
                    true_horizon_actions.append(self.env.sample_action())
            # pdb.set_trace()
            actions = np.concatenate((actions, [true_horizon_actions]), axis=0)  #Add the true action to list of candidate actions
            # actions = np.concatenate([self.true_actions[mpc_step].reshape((1, -1)), actions]) #Add the true action to list of candidate actions

        print(actions.shape)
        if self.use_action_image:
            obs_rep = ptu.from_numpy(np.moveaxis(np.stack([self.env.try_action(action) for action in actions]), 3, 1))
        else:
            obs_rep = ptu.from_numpy(np.moveaxis(obs, 2, 0)).unsqueeze(0).repeat(actions.shape[0], 1, 1, 1) #obs: (D, D, 3) -> (num_actions, 3, D, D)

        pred_obs, obs_latents, obs_latents_recon = self.model_step_batched(obs_rep, ptu.from_numpy(actions))
        # goal_latents (5,128)
        # goal_latents_recon (5, 3, 64, 64)
        # goal_image (1, 3, 64, 64)
        # obs_latents (960, K, 128)
        # obs_latents_recon (960, K, 3, 64, 64)
        # pred_obs (960, 3, 64, 64)
        # actions (960, 13)

        best_pred_obs, best_actions, best_goal_idx, sorted_costs, best_action_idxs = self.cost.best_action(mpc_step, goal_latents,
                                                                           goal_latents_recon,
                                                                           goal_image,
                                                                           obs_latents,
                                                                           obs_latents_recon,
                                                                           pred_obs,
                                                                           actions,
                                                                           cem_step)


        plot_actions = 6
        obs = np.array([self.env.try_step(best_actions[i]) / 255 for i in range(plot_actions)])  # (plot_actions, D, D, 3)
        # obs = np.moveaxis(obs, (0, 1, 2, 3), (0, 3, 1, 2)) #(plot_actions, 3, D, D)
        obs = np.transpose(obs, (0, 3, 1, 2))  # (plot_actions, 3, D, D)
        obs = np.stack([obs, ptu.get_numpy(pred_obs[best_action_idxs[:plot_actions]])])  # (2, plot_actions, 3, D, D)
        caption = np.zeros((2, plot_actions))
        caption[1, :] = ptu.get_numpy(sorted_costs[:plot_actions])

        # First arg should be (h, w, imsize, imsize, 3), caption (h,w)
        plot_multi_image(obs, logger.get_snapshot_dir() + '{}/mpc_best_actions_{}_{}.png'.format(self.logger_prefix_dir,
                                                                                                 mpc_step, cem_step),
                         caption=caption)
        # plot_size = 8
        # full_plot = full_plot[:, :plot_size]
        # caption = np.zeros(full_plot.shape[:2])
        # caption[0, :] = ptu.get_numpy(sorted_costs[:plot_size])
        # caption[1:1 + K, :] = ptu.get_numpy(corresponding_costs[:plot_size])[:, :plot_size]
        #
        # plot_multi_image(ptu.get_numpy(full_plot),
        #                  logger.get_snapshot_dir() + '{}/mpc_pred_{}_{}.png'.format(self.logger_prefix_dir, mpc_step,
        #                                                                             self.cem_step), caption=caption)

        return best_pred_obs, best_actions, best_goal_idx


def main(variant):
    model_file = variant['model_file']


    module_path = get_module_path()

    goal_idxs = [0, 1]

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

    actions_lst = []
    stats = {'mse': 0}

    for i, goal_idx in enumerate(goal_idxs):
        #goal_file = module_path + '/examples/mpc/stage1/manual_constructions/bridge/%d_1.png' % i
        goal_file = module_path + '/examples/mpc/stage3/goals/img_%d.png' % goal_idx
        env_info = np.load(module_path + '/examples/mpc/stage3/goals/env_data.npy')[goal_idx]
        env = BlockPickAndPlaceEnv(num_objects=1, num_colors=None, img_dim=64, include_z=False) #Note num_objects & num_colors do not matter
        env.set_env_info(env_info) #Places the correct blocks in the environment, blocks will also be set in the goal position
        true_actions = env.move_blocks_side() #Moves blocks to the side for mpc, returns true optimal actions

        mpc = MPC(m, env, n_actions=10, mpc_steps=1, true_actions=true_actions,
                  cost_type=variant['cost_type'], filter_goals=False, n_goal_objs=1,
                  logger_prefix_dir='/goal_{}'.format(goal_idx),
                  mpc_style=variant['mpc_style'], cem_steps=3, use_action_image=False, time_horizon=2)
        goal_image = imageio.imread(goal_file)
        mse, actions = mpc.run(goal_image)
        stats['mse'] += mse
        actions_lst.append(actions)

    stats['mse'] /= len(goal_idxs)
    json.dump(stats, open(logger.get_snapshot_dir() + '/stats.json', 'w'))
    np.save(logger.get_snapshot_dir() + '/optimal_actions.npy', np.stack(actions_lst))


#CUDA_VISIBLE_DEVICES=7 python mpc_stage3_v2.py -f /nfs/kun1/users/rishiv/Research/op3_exps/06-10-iodine-blocks-pickplace-multienv-10k/06-10-iodine-blocks-pickplace_multienv_10k_2019_06_10_23_24_47_0000--s-18660/_params.pkl

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--modelfile', type=str, default=None)
    args = parser.parse_args()

    variant = dict(
        algorithm='MPC',
        model_file=args.modelfile,
        cost_type='sum_goal_min_latent_function',  # 'sum_goal_min_latent' 'latent_pixel 'sum_goal_min_latent_function'
        mpc_style='cem', # random_shooting or cem
        model=iodine.imsize64_large_iodine_architecture, #imsize64_large_iodine_architecture
        K=4,
        schedule_kwargs=dict(
            train_T=21,  # Number of steps in single training sequence, change with dataset
            test_T=21,  # Number of steps in single testing sequence, change with dataset
            seed_steps=4,  # Number of seed steps
            schedule_type='curriculum'  # single_step_physics, curriculum
        )
    )

    run_experiment(
        main,
        exp_prefix='mpc_stage3',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )
