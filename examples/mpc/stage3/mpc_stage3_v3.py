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

from examples.mpc.savp_wrapper import SAVP_MODEL

from collections import OrderedDict
from rlkit.util.misc import get_module_path
import pdb
import random

class Cost:
    def __init__(self, logger_prefix_dir, latent_or_subimage='subimage', compare_func='intersect',
                 post_process='raw', aggregate='sum'):
        self.remove_goal_latents = False
        self.logger_prefix_dir = logger_prefix_dir
        self.latent_or_subimage = latent_or_subimage
        self.compare_func = compare_func
        self.post_process = post_process
        self.aggregate = aggregate

    # One function that sums and plots accordingly
    # One function that takes min and plots accordingly
    # Need to specify the following:
    #   Using latent or subimage
    #   Given a pairing, converting to single value (e.g. l2/mse)
    #   Post processing score (e.g. -e^-x)
    #   Taking min or sum

    # goal_latents: (n_goal_latents=K, rep_size)
    # goal_latents_recon: (n_goal_latents=K, 3, 64, 64)
    # goal_image: (1, 3, 64, 64)
    # pred_latents: (n_actions, K, rep_size)
    # pred_latents_recon: (n_actions, K, 3, 64, 64)
    # pred_images: (n_actions, 3, 64, 64)
    def get_action_rankings(self, goal_latents, goal_latents_recon, goal_image, pred_latents,
                    pred_latents_recon, pred_image, image_suffix="", plot_actions=8):
        self.image_suffix = image_suffix
        self.plot_actions = plot_actions
        if self.aggregate == 'sum':
            return self.sum_aggregate(goal_latents, goal_latents_recon, goal_image, pred_latents, pred_latents_recon, pred_image)
        elif self.aggregate == 'min':
            return self.min_aggregate(goal_latents, goal_latents_recon, goal_image, pred_latents, pred_latents_recon, pred_image)
        else:
            raise KeyError

    def get_single_costs(self, goal_latent, goal_latent_recon, pred_latent, pred_latent_recon):
        if self.latent_or_subimage == 'subimage':
            dists = self.compare_subimages(goal_latent_recon, pred_latent_recon)
        elif self.latent_or_subimage == 'latent':
            dists = self.compare_latents(goal_latent, pred_latent)
        else:
            raise ValueError("Invalid latent_or_subimage: {}".format(self.latent_or_subimage))

        costs = self.post_process_func(dists)
        return costs


    def compare_latents(self, goal_latent, pred_latent):
        if self.compare_func == 'mse':
            return self.mse(goal_latent.view(1, 1, -1), pred_latent)
        else:
            raise KeyError

    def compare_subimages(self, goal_latent_recon, pred_latent_recon):
        if self.compare_func == 'mse':
            return self.image_mse(goal_latent_recon.view((1, 1, 3, 64, 64)), pred_latent_recon)
        elif self.compare_func == 'intersect':
            return self.image_intersect(goal_latent_recon.view((1, 1, 3, 64, 64)),
                                        pred_latent_recon)
        else:
            raise KeyError

    def post_process_func(self, dists):
        if self.post_process == 'raw':
            return dists
        elif self.post_process == 'negative_exp':
            return -torch.exp(-dists)
        else:
            raise KeyError

    # goal_latents: (n_goal_latents=K, rep_size)
    # goal_latents_recon: (n_goal_latents=K, 3, 64, 64)
    # goal_image: (1, 3, 64, 64)
    # pred_latents: (n_actions, K, rep_size)
    # pred_latents_recon: (n_actions, K, 3, 64, 64)
    # pred_images: (n_actions, 3, 64, 64)
    def sum_aggregate(self, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_images):

        n_goal_latents = goal_latents.shape[0] #Note, this should equal K if we did not filter anything out
        # Compare against each goal latent
        costs = []  #(n_goal_latents, n_actions)
        latent_idxs = [] # (n_goal_latents, n_actions), [a,b] is an index corresponding to a latent
        for i in range(n_goal_latents): #Going through all n_goal_latents goal latents
            # pdb.set_trace()
            single_costs = self.get_single_costs(goal_latents[i], goal_latents_recon[i], pred_latents, pred_latents_recon)
            min_costs, latent_idx = single_costs.min(-1)  # take min among K, size is (n_actions)

            costs.append(min_costs)
            latent_idxs.append(latent_idx)

        costs = torch.stack(costs) # (n_goal_latents, n_actions)
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions)

        #Sort by sum cost
        #Image contains the following: Pred_images, goal_latent_reconstructions, and
        # corresponding pred_latent_reconstructions
        #For every latent in goal latents, find corresponding predicted one (this is in latent_idxs)
        #  Should have something that is (K, num_actions) -> x[a,b] is index for pred_latents_recon
        sorted_costs, best_action_idxs = costs.sum(0).sort()

        if self.plot_actions:
            sorted_pred_images = pred_images[best_action_idxs]

            corresponding_pred_latent_recons = []
            for i in range(n_goal_latents):
                tmp = pred_latents_recon[best_action_idxs, latent_idxs[i, best_action_idxs]] #(n_actions, 3, 64, 64)
                corresponding_pred_latent_recons.append(tmp)
            corresponding_pred_latent_recons = torch.stack(corresponding_pred_latent_recons) #(n_goal_latents, n_actions, 3, 64, 64)
            corresponding_costs = costs[:, best_action_idxs]

            full_plot = torch.cat([sorted_pred_images.unsqueeze(0), # (1, n_actions, 3, 64, 64)
                                   corresponding_pred_latent_recons, # (n_goal_latents, n_actions, 3, 64, 64)
                                   ], 0)
            plot_size = self.plot_actions
            full_plot = full_plot[:, :plot_size]

            #Add goal latents
            tmp = torch.cat([goal_image, goal_latents_recon], dim=0).unsqueeze(1) #(n_goal_latents+1, 1, 3, 64, 64)
            full_plot = torch.cat([tmp, full_plot], dim=1)

            #Add captions
            caption = np.zeros(full_plot.shape[:2])
            caption[0, 1:] = ptu.get_numpy(sorted_costs[:plot_size])
            caption[1:1+n_goal_latents, 1:] = ptu.get_numpy(corresponding_costs[:plot_size])[:,:plot_size]

            plot_multi_image(ptu.get_numpy(full_plot),
                             logger.get_snapshot_dir() + '{}/cost_{}.png'.format(self.logger_prefix_dir, self.image_suffix), caption=caption)
        return ptu.get_numpy(sorted_costs), ptu.get_numpy(best_action_idxs), np.zeros(len(sorted_costs))

    def min_aggregate(self, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_image):

        n_goal_latents = goal_latents.shape[0]
        num_actions = pred_latents.shape[0]
        # Compare against each goal latent
        costs = []  # (n_goal_latents, n_actions)
        latent_idxs = []  # (n_goal_latents, n_actions), [a,b] is an index corresponding to a latent
        for i in range(n_goal_latents):  # Going through all n_goal_latents goal latents
            single_costs = self.get_single_costs(goal_latents[i], goal_latents_recon[i], pred_latents, pred_latents_recon) #(K, n_actions)
            min_costs, latent_idx = single_costs.min(-1)  # take min among K, size is (n_actions)

            costs.append(min_costs)
            latent_idxs.append(latent_idx)

        costs = torch.stack(costs) # (n_goal_latents, n_actions)
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions)

        #Sort by sum cost
        #Image contains the following: Pred_images, goal_latent_reconstructions, and
        # corresponding pred_latent_reconstructions
        #For every latent in goal latents, find corresponding predicted one (this is in latent_idxs)
        #  Should have something that is (K, num_actions) -> x[a,b] is index for pred_latents_recon
        min_costs, min_goal_latent_idx = costs.min(0) #(num_actions)
        sorted_costs, best_action_idxs = min_costs.sort() #(num_actions)

        if self.plot_actions:
            sorted_pred_images = pred_image[best_action_idxs]
            corresponding_pred_latent_recons = []
            for i in range(n_goal_latents):
                tmp = pred_latents_recon[best_action_idxs, latent_idxs[i, best_action_idxs]] #(n_actions, 3, 64, 64)
                corresponding_pred_latent_recons.append(tmp)
            corresponding_pred_latent_recons = torch.stack(corresponding_pred_latent_recons) #(n_goal_latents, n_actions, 3, 64, 64)
            corresponding_costs = costs[:, best_action_idxs] # (n_goal_latents, n_actions)

            # pdb.set_trace()
            min_corresponding_latent_recon = pred_latents_recon[best_action_idxs, latent_idxs[min_goal_latent_idx[best_action_idxs], best_action_idxs]] #(n_actions, 3, 64, 64)

            # pdb.set_trace()

            full_plot = torch.cat([sorted_pred_images.unsqueeze(0), # (1, n_actions, 3, 64, 64)
                                   corresponding_pred_latent_recons, # (n_goal_latents=K, n_actions, 3, 64, 64)
                                   min_corresponding_latent_recon.unsqueeze(0) #(1, n_actions, 3, 64, 64)
                                   ], 0) # (n_goal_latents+2, n_actions, 3, 64, 64)
            plot_size = self.plot_actions
            full_plot = full_plot[:, :plot_size] # (n_goal_latents+2, plot_size, 3, 64, 64)

            # Add goal latents
            tmp = torch.cat([goal_image, goal_latents_recon, goal_image], dim=0).unsqueeze(1)  # (n_goal_latents+2, 1, 3, 64, 64)
            full_plot = torch.cat([tmp, full_plot], dim=1) # (n_goal_latents+2, plot_size+1, 3, 64, 64)

            #Add captions
            caption = np.zeros(full_plot.shape[:2])
            caption[0, 1:] = ptu.get_numpy(sorted_costs[:plot_size])
            caption[1:1+n_goal_latents, 1:] = ptu.get_numpy(corresponding_costs[:plot_size])[:,:plot_size]

            plot_multi_image(ptu.get_numpy(full_plot),
                             logger.get_snapshot_dir() + '{}/mpc_pred_{}.png'.format(self.logger_prefix_dir,
                                                                                     self.image_suffix), caption=caption)
        return ptu.get_numpy(sorted_costs), ptu.get_numpy(best_action_idxs), ptu.get_numpy(min_goal_latent_idx)

    def mse(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

    def image_intersect(self, im1, im2):
        # im1, im2 are (*, 3, D, D)
        # Note: * dimensions may not be equal between im1, im2 so automatically broadcast over them
        #import pdb; pdb.set_trace()
        m1 = (im1 > 0.01).float()
        m2 = (im2 > 0.01).float()
        intersect = (m1 * m2).float()
        intersect.sum((-3, -2, -1))
        union = m1.sum((-3, -2, -1)) + m2.sum((-3, -2, -1))
        #iou = intersect.sum((-3, -2, -1)) / union

        threshold_intersect = (intersect * (torch.pow(im1 - im2, 2) < 0.08).float()).sum((-3,
                                                                                              -2,
                                                                                         -1))

        iou = threshold_intersect / union

        #import pdb; pdb.set_trace()

        return (1 - iou)
        # the last

    def image_mse(self, im1, im2):
        # im1, im2 are (*, 3, D, D)
        # Note: * dimensions may not be equal between im1, im2 so automatically broadcast over them

        return torch.pow(im1 - im2, 2).mean(-1).mean(-1).mean(-1) #Takes means across the last
        # dimensions
        # (3,
        # D, D)

class MPC:
    def __init__(self, model, env, n_actions, mpc_steps,
                 n_goal_objs=3,
                 cost_type='latent_pixel',
                 filter_goals=False,
                 true_actions=None,
                 logger_prefix_dir=None,
                 mpc_style="random_shooting",  # options are random_shooting, cem
                 cem_steps=2,
                 true_data=None,
                 use_action_image=True, # True for stage 1, False for stage 3
                 time_horizon=1, #How many steps into the future to do per step
                 actions_per_step=1 #How many actions to take per step. Note this needs to be <= time_horizon
                 ):
        self.model = model
        self.env = env
        self.n_actions = n_actions
        self.mpc_steps = mpc_steps
        self.cost_type = cost_type
        self.filter_goals = filter_goals
        self.cost = Cost(logger_prefix_dir)
        self.true_actions = true_actions
        self.n_goal_objs = n_goal_objs
        self.mpc_style = mpc_style
        self.cem_steps = cem_steps
        if logger_prefix_dir is not None:
            os.mkdir(logger.get_snapshot_dir() + logger_prefix_dir)
        self.logger_prefix_dir = logger_prefix_dir
        self.use_action_image = use_action_image
        self.time_horizon = time_horizon

        if actions_per_step > time_horizon:
            raise ValueError("actions_per_step ({}) should be <= time_horizon ({})".format(actions_per_step, time_horizon))
        self.actions_per_step = actions_per_step
        self.true_data = true_data

    def filter_goal_latents(self, goal_latents, goal_latents_mask, goal_latents_recon):
        # Keep top goal latents with highest mask area except first
        # goal_latents: (K, 128)
        # goal_latents_mask: (K, 64, 64)
        # goal_latents_recon: (K, 3, 64, 64)

        # pdb.set_trace()
        n_goals = self.n_goal_objs
        goal_latents_mask[goal_latents_mask < 0.5] = 0
        vals, sorted_idx = torch.sort(goal_latents_mask.mean((1, 2)), descending=True)

        save_image(goal_latents_mask[sorted_idx].unsqueeze(1).repeat(1, 3, 1, 1),
                   logger.get_snapshot_dir() + '{}/goal_masks.png'.format(self.logger_prefix_dir))

        keep = sorted_idx[2:2 + n_goals]
        goal_latents = goal_latents[keep]
        goal_latents_recon = goal_latents_recon[keep]

        # save_image(goal_latents_recon,
        #            logger.get_snapshot_dir() + '{}/goal_latents_recon.png'.format(self.logger_prefix_dir))

        return goal_latents, goal_latents_recon

    def remove_idx(self, array, idx):
        return torch.stack([array[i] for i in set(range(array.shape[0])) - set([idx])])

    def run(self, goal_image):
        goal_image_tensor = ptu.from_numpy(np.moveaxis(goal_image, 2, 0)).unsqueeze(0).float()[:, :3] / 255.  # (1, 3, imsize, imsize)

        rec_goal_image, goal_latents, goal_latents_recon, goal_latents_mask = self.model.refine(
            goal_image_tensor,
            hidden_state=None,
            plot_latents=False)  # (K, rep_size)
        # pdb.set_trace()

        # Keep top 4 goal latents with greatest mask area excluding 1st (background)
        if self.filter_goals:
            goal_latents, goal_latents_recon = self.filter_goal_latents(goal_latents, goal_latents_mask, goal_latents_recon)
        save_image(goal_latents_recon, logger.get_snapshot_dir() + '{}/goal_latents_recon.png'.format(self.logger_prefix_dir))

        #true_actions = self.env.move_blocks_side()
        #self.true_actions = true_actions
        obs = self.env.get_observation()/255
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

        imageio.imsave(logger.get_snapshot_dir() + '{}/initial_image.png'.format(self.logger_prefix_dir), obs*255)
        obs_lst = [np.moveaxis(goal_image.astype(np.float32) / 255., 2, 0), np.moveaxis(obs, 2, 0)]

        initial_pred_obs = self.model.refine(ptu.from_numpy(np.moveaxis(obs, 2, 0)), hidden_state=None, plot_latents=False)[0]
        pred_obs_lst = [ptu.get_numpy(rec_goal_image), ptu.get_numpy(initial_pred_obs)]
        full_obs_list = []

        chosen_actions = []
        best_accuracy = 0

        # print("self.mpc_steps: {}".format(self.mpc_steps))
        for mpc_step in range(self.mpc_steps):
            pred_obs, actions, goal_idx = self.step_mpc(obs, goal_latents, goal_image_tensor, mpc_step, goal_latents_recon)
            # pdb.set_trace()
            chosen_actions.append(actions[0])

            #Add list individual observations
            tmp = [obs]
            tmp.extend([self.env.try_step(actions[:i+1])/255 for i in range(self.time_horizon)])
            full_obs_list.append(tmp)

            for i in range(self.actions_per_step):
                obs = self.env.step(actions[i])/255

            # obs = self.env.step(actions[0])/255 #Step the environment
            obs_lst.append(np.moveaxis(obs, 2, 0))
            pred_obs_lst.append(pred_obs)
            # remove matching goal latent from goal latents
            if self.cost.remove_goal_latents:
                goal_latents = self.remove_idx(goal_latents, goal_idx)
                goal_latents_recon = self.remove_idx(goal_latents_recon, goal_idx)
                #print(goal_latents.shape)

            if self.time_horizon > 1:
                self.time_horizon -= 1

            accuracy = self.env.compute_accuracy(self.true_data, threshold=0.25)
            best_accuracy = max(accuracy, best_accuracy)

        save_image(ptu.from_numpy(np.stack(obs_lst + pred_obs_lst)),
                   logger.get_snapshot_dir() + '{}/mpc.png'.format(self.logger_prefix_dir), nrow=len(obs_lst))

        #full_obs_list = np.stack(full_obs_list) #(mpc_step, Th, 64, 64, 3)
        #plot_multi_image(full_obs_list, logger.get_snapshot_dir() + '{
        # }/mpc_step_by_step.png'.format(self.logger_prefix_dir))

        # Compare final obs to goal obs
        mse = np.square(ptu.get_numpy(goal_image_tensor.squeeze().permute(1, 2, 0)) - obs).mean()
        #accuracy = self.env.compute_accuracy(self.true_data)

        return best_accuracy, np.stack(chosen_actions)

    #actions: (n_actions, T, A)
    #obs: (n_actions, 3, D, D)
    # NOTE: Obs must be between range (0,1)!
    # def model_step_batched(self, obs, actions, bs=4):
    #     # Handle large obs in batches
    #     n_batches = int(np.ceil(obs.shape[0] / float(bs)))
    #     outputs = [[], [], []]
    #
    #     # old_env_info = self.env.get_env_info()
    #     # real_obs = []
    #     # for action in ptu.get_numpy(actions):
    #     #     self.env.set_env_info(old_env_info)
    #     #     actual_obs = self.env.step(action)
    #     #     real_obs.append(actual_obs)
    #     # self.env.set_env_info(old_env_info)
    #
    #
    #     for i in range(n_batches):
    #         start_idx = i * bs
    #         end_idx = min(start_idx + bs, obs.shape[0])
    #         actions_batch = actions[start_idx:end_idx] if not self.use_action_image else None
    #
    #         pred_obs, obs_latents, obs_latents_recon = self.model.step(obs[start_idx:end_idx], actions_batch, plot_latents=False)
    #         outputs[0].append(pred_obs)
    #         outputs[1].append(obs_latents)
    #         outputs[2].append(obs_latents_recon)
    #
    #     # save_image(ptu.from_numpy(np.concatenate([np.moveaxis(np.stack(real_obs)/255., 3, 1), ptu.get_numpy(pred_obs)])),
    #     #            logger.get_snapshot_dir() + '{}/pred_vs_real.png'.format(self.logger_prefix_dir), nrow=7)
    #
    #     return torch.cat(outputs[0]), torch.cat(outputs[1]), torch.cat(outputs[2])

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
        #print("self.cem_steps: {}".format(self.cem_steps))
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

        #print(actions.shape)
        if self.use_action_image:
            obs_rep = ptu.from_numpy(np.moveaxis(np.stack([self.env.try_action(action) for action in actions]), 3, 1))
        else:
            obs_rep = ptu.from_numpy(np.moveaxis(obs, 2, 0)).unsqueeze(0).repeat(actions.shape[0], 1, 1, 1) #obs: (D, D, 3) -> (num_actions, 3, D, D)

        pred_obs, obs_latents, obs_latents_recon = self.model.step_batched(obs_rep,
                                                                           ptu.from_numpy(
                                                                               actions), bs=10) #self.model_step_batched(obs_rep, ptu.from_numpy(actions))
        # goal_latents (5,128)
        # goal_latents_recon (5, 3, 64, 64)
        # goal_image (1, 3, 64, 64)
        # obs_latents (960, K, 128)
        # obs_latents_recon (960, K, 3, 64, 64)
        # pred_obs (960, 3, 64, 64)
        # actions (960, 13)

        #Note: All the below three are numpy arrays
        sorted_costs, best_action_idxs, goal_latent_idxs = self.cost.get_action_rankings(goal_latents, goal_latents_recon, goal_image,
                                                                                         obs_latents, obs_latents_recon, pred_obs,
                                                                                         "mpc_{}_{}".format(mpc_step, cem_step))

        sorted_actions = actions[best_action_idxs]
        plot_actions = 6
        best_obs = np.array([self.env.try_step(sorted_actions[i]) / 255 for i in range(plot_actions)])  # (plot_actions, D, D, 3)
        # obs = np.moveaxis(obs, (0, 1, 2, 3), (0, 3, 1, 2)) #(plot_actions, 3, D, D)
        best_obs = np.transpose(best_obs, (0, 3, 1, 2))  # (plot_actions, 3, D, D)
        best_obs = np.stack([best_obs, ptu.get_numpy(pred_obs[best_action_idxs[:plot_actions]])])  # (2, plot_actions, 3, D, D)
        caption = np.zeros((2, plot_actions+1))
        caption[1, 1:] = sorted_costs[1:plot_actions+1]

        # First arg should be (h, w, imsize, imsize, 3) or (h, w, 3, imsize, imsize), caption (h,w)
        full_plot = np.concatenate([[np.expand_dims(np.moveaxis(obs, 2, 0),0)]*2, best_obs], axis=1) # (2, plot_actions+1, 3, D, D)
        plot_multi_image(full_plot, logger.get_snapshot_dir() + '{}/mpc_best_actions_{}_{}.png'.format(self.logger_prefix_dir,
                                                                                                 mpc_step, cem_step), caption=caption)

        # return best_pred_obs, best_actions, best_goal_idx
        # pdb.set_trace()
        return ptu.get_numpy(pred_obs[best_action_idxs[0]]), sorted_actions, goal_latent_idxs[best_action_idxs[0]]


def load_model(variant):
    if variant['model'] == 'savp':
        time_horizon = variant['mpc_args']['time_horizon']
        m = SAVP_MODEL('/home/jcoreyes/objects/baseline/logs/pickplace_multienv_10k/ours_savp'
                       '/',
                       'model-500000', 0,
                       batch_size=10, time_horizon=time_horizon)
    else:
        model_file = variant['model_file']
        m = iodine.create_model(variant, action_dim=4)
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

# def load_model(variant, action_size):
#     if variant['model'] == 'savp':
#         time_horizon = variant['mpc_args']['time_horizon']
#         m = SAVP_MODEL('/nfs/kun1/users/rishiv/Research/baseline/logs/pickplace_multienv_10k/ours_savp/', 'model-500000', 0,
#                        batch_size=20, time_horizon=time_horizon)
#     else:
#         model_file = variant['model_file']
#
#         if variant['model_type'] == 'next_step':
#             variant['model']['refine_args']['added_fc_input_size'] = action_size
#         elif variant['model_type'] == 'static':
#             action_size = 0
#         m = iodine.create_model(variant, action_dim=action_size)
#         state_dict = torch.load(model_file)
#         # pdb.set_trace()
#
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k
#             if 'module.' in k:
#                 name = k[7:]  # remove 'module.' of dataparallel
#             new_state_dict[name] = v
#         m.load_state_dict(new_state_dict)
#         m.cuda()
#         m.set_eval_mode(True)
#     return m


def main(variant):
    seed = int(variant['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    module_path = get_module_path()

    m = load_model(variant)

    goal_idxs = list(range(0, 20))
    actions_lst = []
    stats = {'accuracy': 0}

    goal_folder = module_path + '/examples/mpc/stage3/goals/objects_{}/'.format(variant['number_goal_objects'])

    for i, goal_idx in enumerate(goal_idxs):
        #goal_file = module_path + '/examples/mpc/stage1/manual_constructions/bridge/%d_1.png' % i
        goal_file = goal_folder + 'img_{}.png'.format(goal_idx)
        env_info = np.load(goal_folder + 'env_data.npy')[goal_idx]
        env = BlockPickAndPlaceEnv(num_objects=1, num_colors=None, img_dim=64, include_z=False) #Note num_objects & num_colors do not matter
        env.set_env_info(env_info) #Places the correct blocks in the environment, blocks will also be set in the goal position
        true_actions = env.move_blocks_side()  # Moves blocks to the side for mpc, returns true optimal actions

        if variant['mpc_args']['true_actions']:
            variant['mpc_args']['true_actions'] = true_actions
        else:
            variant['mpc_args']['true_actions'] = None


        # mpc = MPC(m, env, n_actions=20, mpc_steps=3, true_actions=None,
        #           cost_type=variant['cost_type'], filter_goals=False, n_goal_objs=2,
        #           logger_prefix_dir='/goal_{}'.format(goal_idx),
        #           mpc_style=variant['mpc_style'], cem_steps=3, use_action_image=False, time_horizon=2, actions_per_step=2)
        mpc = MPC(m, env, logger_prefix_dir='/goal_{}'.format(goal_idx),
                  true_data=env_info, **variant[
            'mpc_args'])

        goal_image = imageio.imread(goal_file)
        accuracy, actions = mpc.run(goal_image)
        stats['accuracy'] += accuracy
        actions_lst.append(actions)
        print("goal_idx %d accuracy: %f" % (i, stats['accuracy'] / float((i+1))))
        np.save(logger.get_snapshot_dir() + '/optimal_actions.npy', np.stack(actions_lst))

    stats['accuracy'] /= len(goal_idxs)
    json.dump(stats, open(logger.get_snapshot_dir() + '/stats.json', 'w'))
    # np.save(logger.get_snapshot_dir() + '/optimal_actions.npy', np.stack(actions_lst))


#CUDA_VISIBLE_DEVICES=3 python mpc_stage3_v3.py -f /nfs/kun1/users/rishiv/Research/op3_exps/06-10-iodine-blocks-pickplace-multienv-10k/06-10-iodine-blocks-pickplace_multienv_10k_2019_06_10_23_24_47_0000--s-18660/_params.pkl
#CUDA_VISIBLE_DEVICES=3 python mpc_stage3_v3.py -f /nfs/kun1/users/rishiv/Research/op3_exps/06-26-iodine-blocks-pickplace-multienv-10k-k1/06-26-iodine-blocks-pickplace_multienv_10k-k1_2019_06_26_03_50_18_0000--s-15069/_params.pkl
#CUDA_VISIBLE_DEVICES=3 python mpc_stage3_v3.py -f /nfs/kun1/users/rishiv/Research/op3_exps/06-26-iodine-blocks-pickplace-multienv-10k-mlp/06-26-iodine-blocks-pickplace_multienv_10k-mlp_2019_06_26_22_54_03_0000--s-16393/_params.pkl

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--modelfile', type=str, default=None)
    args = parser.parse_args()

    num_obs = 2 #TODO: Change

    variant = dict(
        algorithm='SAVP',
        number_goal_objects=num_obs,
        model_file=args.modelfile,
        # cost_type='sum_goal_min_latent_function',  # 'sum_goal_min_latent' 'latent_pixel 'sum_goal_min_latent_function'
        # mpc_style='cem', # random_shooting or cem
        model='savp', #iodine.imsize64_large_iodine_architecture_multistep_physics, #'savp', ' \
                                                                                   #'#iodine.imsize64_large_iodine_architecture_multistep_physics, #imsize64_large_iodine_architecture 'savp',
        K=4,
        schedule_kwargs=dict(
            train_T=21,  # Number of steps in single training sequence, change with dataset
            test_T=21,  # Number of steps in single testing sequence, change with dataset
            seed_steps=4,  # Number of seed steps
            schedule_type='curriculum'  # single_step_physics, curriculum
        ),
        mpc_args=dict(
            n_actions=500,
            mpc_steps=2,
            time_horizon=2,
            actions_per_step=1,
            cem_steps=1,
            use_action_image=False,
            mpc_style='cem',
            n_goal_objs=num_obs,
            filter_goals=True,
            true_actions=False,
        )
    )

    run_experiment(
        main,
        exp_prefix='mpc_stage3_objects{}-{}'.format(variant['number_goal_objects'],
                                                    variant['model']),
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=None,
    )
