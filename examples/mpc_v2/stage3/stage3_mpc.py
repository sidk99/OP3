import matplotlib
matplotlib.use('Agg')

from rlkit.torch import pytorch_util as ptu
import rlkit.torch.iodine.visualizer as visualizer
from rlkit.util.plot import plot_multi_image
from torchvision.utils import save_image
import torch
import numpy as np
from argparse import ArgumentParser
from rlkit.launchers.launcher_util import run_experiment

import rlkit.torch.iodine.iodine_v2 as iodine_v2
from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
from rlkit.util.misc import get_module_path
from examples.mpc_v2.stage3.saved_models.model_parameters_info import params_to_info

from collections import OrderedDict
import shutil
import pickle
import json
import imageio
import os
import pdb
# from abc import ABC, abstractmethod

##############Cost Class ############
class Cost:
    def __init__(self, logging_directory, core_type="subimage", compare_func='mse', post_process='raw', aggregate='sum'):
        self.logging_directory = logging_directory
        self.core_type = core_type
        if core_type not in ["subimage", "final_recon", "latent"]:
            raise ValueError("Invalid value of core_type: {}".format(core_type))
        self.compare_func = compare_func
        self.post_process = post_process
        self.aggregate = aggregate


    # Inputs: goal_latents (n_goal_latents=K,R), goal_latents_recon (n_goal_latents=K,3,64,64)
    # goal_image (1,3,64,64), pred_latents (n_actions,K,R), pred_latents_recon (n_actions,K,3,64,64)
    # pred_images (n_actions,3,64,64)
    def get_action_rankings(self, goal_latents, goal_latents_recon, goal_image, pred_latents,
                    pred_latents_recon, pred_image, image_suffix="", plot_actions=8):
        self.image_suffix = image_suffix
        self.plot_actions = plot_actions

        if self.core_type == "final_recon": # Pretend that we only did K=1
            goal_latents_recon = goal_image  # (K=1,3,64,64)
            pred_latents_recon = pred_image.unsqueeze(1)  # (n_actions,K=1,3,64,64)

        if self.aggregate == 'sum':
            return self.sum_aggregate(goal_latents, goal_latents_recon, goal_image, pred_latents, pred_latents_recon, pred_image)
        elif self.aggregate == 'min':
            return self.min_aggregate(goal_latents, goal_latents_recon, goal_image, pred_latents, pred_latents_recon, pred_image)
        else:
            raise KeyError

    def get_single_costs(self, goal_latent, goal_latent_recon, pred_latent, pred_latent_recon):
        if self.core_type == 'subimage' or self.core_type == "final_recon":
            dists = self.compare_subimages(goal_latent_recon, pred_latent_recon)
        elif self.core_type == 'latent':
            dists = self.compare_latents(goal_latent, pred_latent)
        else:
            raise ValueError("Invalid core_type: {}".format(self.core_type))

        costs = self.post_process_func(dists)
        return costs


    def compare_latents(self, goal_latent, pred_latent):
        if self.compare_func == 'mse':
            return self.mse(goal_latent.view(1, 1, -1), pred_latent)
        else:
            raise KeyError

    # Input: goal_latent_recon (3,D,D),  pred_latent_recon (Na,K,3,D,D)
    def compare_subimages(self, goal_latent_recon, pred_latent_recon):
        if self.compare_func == 'mse':
            return self.image_mse(goal_latent_recon.view((1, 1, 3, 64, 64)), pred_latent_recon)
        elif self.compare_func == 'psuedo_intersect':
            # pdb.set_trace()
            m1 = (goal_latent_recon > 0.01).float()  # (3,D,D)
            m2 = (pred_latent_recon > 0.01).float()  # (Na,K,3,D,D)
            intersect = (m1 * m2).float()  # (Na,K,3,D,D)
            intersect.sum((-3, -2, -1))    # (Na,K)
            union = m1.sum((-3, -2, -1)) + m2.sum((-3, -2, -1))  # (Na,K)
            threshold_intersect = (intersect * (torch.pow(goal_latent_recon - pred_latent_recon, 2) < 0.08).float()).sum((-3, -2, -1))  # (Na,K)
            iou = threshold_intersect / union  # (Na,K)
            return 1 - iou
        else:
            raise KeyError


    def post_process_func(self, dists):
        if self.post_process == 'raw':
            return dists
        elif self.post_process == 'negative_exp':
            return -torch.exp(-dists)
        else:
            raise KeyError

    # Inputs: goal_latents (n_goal_latents=K,R), goal_latents_recon (n_goal_latents=K,3,64,64)
    # goal_image (1,3,64,64), pred_latents (n_actions,K,R), pred_latents_recon (n_actions,K,3,64,64)
    # pred_images (n_actions,3,64,64)
    def sum_aggregate(self, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_images):

        n_goal_latents = goal_latents_recon.shape[0] #Note, this should equal K if we did not filter anything out
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
                             '{}/{}.png'.format(self.logging_directory, self.image_suffix), caption=caption)

        return ptu.get_numpy(sorted_costs), ptu.get_numpy(best_action_idxs), np.zeros(len(sorted_costs))

    def min_aggregate(self, goal_latents, goal_latents_recon, goal_image,
                            pred_latents, pred_latents_recon, pred_image):

        n_goal_latents = goal_latents_recon.shape[0]
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
            caption[1:1 + n_goal_latents, 1:] = ptu.get_numpy(corresponding_costs[:, :plot_size])
            # caption[1:1+n_goal_latents, 1:] = ptu.get_numpy(corresponding_costs[:plot_size])[:,:plot_size]

            plot_multi_image(ptu.get_numpy(full_plot),
                             '{}/mpc_pred_{}.png'.format(self.logging_directory, self.image_suffix), caption=caption)


        return ptu.get_numpy(sorted_costs), ptu.get_numpy(best_action_idxs), ptu.get_numpy(min_goal_latent_idx)

    def mse(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

    def image_mse(self, im1, im2):
        # im1, im2 are (*, 3, D, D)
        # Note: * dimensions may not be equal between im1, im2 so automatically broadcast over them
        return torch.pow(im1 - im2, 2).mean((-1, -2, -3)) #Takes means across the last dimensions (3, D, D)

    # # Input: actions (B,A) np,  pred_recons (B,3,D,D)
    # def plot_action_errors(self, env, actions, pred_recons):
    #     errors = env.get_action_error(actions)  # (B) np
    #
    #     full_plot = pred_recons.view([5, -1] + list(pred_recons.shape)) # (5,B//5,3,D,D)
    #     caption = np.reshape(errors, (5, -1))  # (5,B//5) np
    #     plot_multi_image(ptu.get_numpy(full_plot),
    #                      '{}/mpc_pred_{}_errors.png'.format(self.logging_directory, self.image_suffix), caption=caption)



#############Action Selection Class#########
####Stage3 Specific
class Stage3_CEM:
    def __init__(self, logging_dir, cem_steps, num_samples, time_horizon, score_actions_class):
        self.logging_dir = logging_dir
        self.cem_steps = cem_steps
        self.num_samples = num_samples
        self.time_horizon = time_horizon
        self.score_actions_class = score_actions_class
        self.env = None
        self.goal_info = None

    def select_action(self, goal_info, initial_hidden_state, env, model, logging_suffix):
        self.env = env
        self.goal_info = goal_info
        self.initial_hidden_state = initial_hidden_state
        return self._cem(model, logging_suffix)

    # Inputs: actions (B,T,A),  model
    # Outputs: Index of actions based off sorted costs (B), paired goal latent (for removal) (B), final_recons (B,3,D,D)
    def _random_shooting(self, actions, model, image_suffix):
        # Like internal_inference except initial_hidden_state might only contain one state while obs/actions contain (B,*)
        # Inputs: obs (B,T1,3,D,D) or None, actions (B,T2,A) or None, initial_hidden_state or None, schedule (T3)
        #   Note: Assume that initial_hidden_state has entries of size (B=1,*)


        goal_info = self.goal_info
        schedule = np.array([1]*actions.shape[1])
        actions = ptu.from_numpy(actions)
        predicted_info = model.batch_internal_inference(obs=None, actions=actions, initial_hidden_state=self.initial_hidden_state,
                                                          schedule=schedule, figure_path=None)
        # pdb.set_trace()
        sorted_costs, best_actions_indices, goal_latent_indices = self.score_actions_class.get_action_rankings(
            goal_info["state"]["post"]["samples"][0], goal_info["sub_images"][0], goal_info["goal_image"],
            predicted_info["state"]["post"]["samples"], predicted_info["sub_images"], predicted_info["final_recon"],
            image_suffix = image_suffix)
        # Inputs: goal_latents (n_goal_latents=K,R), goal_latents_recon (n_goal_latents=K,3,64,64)
        # goal_image (1,3,64,64), pred_latents (n_actions,K,R), pred_latents_recon (n_actions,K,3,64,64)
        # pred_images (n_actions,3,64,64)

        num_plot_actions = 20
        self.plot_action_errors(self.env, actions[best_actions_indices][:num_plot_actions, 0],
                                predicted_info["final_recon"][best_actions_indices][:num_plot_actions],
                                image_suffix+"_action_errors")
        return best_actions_indices, goal_latent_indices, predicted_info["final_recon"]

    # Inputs: N/A
    # Outputs: actions (B,T,A)
    def _get_initial_actions(self):
        actions = []
        for i in range(self.time_horizon):
            actions.append(np.stack([self.env.sample_action() for _ in range(self.num_samples)]))  # (T,B,A)
        actions = np.array(actions).transpose((1, 0, 2))  # (B,T,A)
        return np.array(actions)

    # Input: model, logging_suffix (Str)
    # Output: best actions (T,A), corresponding latent (Sc), corresponding predicted reconstructions (3,D,D)
    def _cem(self, model, logging_suffix):
        actions = self._get_initial_actions()  # (B,T,A)
        filter_cutoff = int(self.num_samples * 0.1)  # F

        for i in range(self.cem_steps):
            best_actions_indices, goal_latent_indices, pred_recons = self._random_shooting(actions, model, "action_{}_{}".format(logging_suffix, i))  # (B)
            sorted_actions = actions[best_actions_indices]  # (B,T,A)
            best_actions = sorted_actions[:filter_cutoff]  # (F,T,A)
            mean = best_actions.mean(0)  # (T,A)
            std = best_actions.std(0) + 0.05  # (T,A),  +0.05 is added as a minimum std deviation (note pick threshold is 0.2)
            actions = self.env.sample_multiple_action_gaussian(mean, std, self.num_samples)  # (B,T,A)
            print("Step {}".format(i))

        best_action_index = best_actions_indices[0]
        return np.array(best_actions[0]), goal_latent_indices[best_action_index], pred_recons[best_action_index]

    # Input: actions (B,A),  pred_recons (B,3,D,D)
    def plot_action_errors(self, env, actions, pred_recons, file_name):
        errors = env.get_action_error(ptu.get_numpy(actions))  # (B) np

        full_plot = pred_recons.view([5, -1] + list(pred_recons.shape[1:]))  # (5,B//5,3,D,D)
        caption = np.reshape(errors, (5, -1))  # (5,B//5) np
        plot_multi_image(ptu.get_numpy(full_plot), '{}/{}.png'.format(self.logging_dir, file_name), caption=caption)


########Process env functions########
#Input: env_obs (D,D,3) or (T,D,D,3), values between 0-255, numpy array
#Output: (1,3,D,D), values between 0-1, torch
def process_env_obs(env_obs):
    if len(env_obs.shape) == 3: #(D,D,3) numpy
        env_obs = np.expand_dims(env_obs, 0) #(T=1,D,D,3), numpy
    return ptu.from_numpy(np.moveaxis(env_obs, 3, 1))/255

# Input: env_obs (A) or (T,A), numpy array
def process_env_actions(env_actions):
    if len(env_actions.shape) == 1: #(A) numpy
        env_actions = np.expand_dims(env_actions, 0) #(T=1,A), numpy
    return ptu.from_numpy(env_actions)


##############MPC Class ############
class Stage3_MPC:
    def __init__(self, model, logging_dir):
        self.model = model
        self.logging_dir = logging_dir

        if logging_dir is not None:
            if not os.path.exists(logging_dir):
                os.mkdir(logging_dir)

    # Inputs: goal_image (3,D,D), env, initial_obs (T1,3,D,D), initial_actions (T1-1,A), action_selection_class, T
    # Assume all images are between 0 & 1 and are tensors
    def run_plan(self, goal_image, env, initial_obs, initial_actions, action_selection_class,
                 num_actions_to_take, planning_horizon, true_data, filter_goal_image=None):
        #Goal inference
        self.env = env
        goal_info = self.goal_inference(goal_image, filter_goal_image)

        #State acquisition
        cur_state_and_other_info = self.state_acquisition(initial_obs, initial_actions)
        initial_recon = cur_state_and_other_info["final_recon"]

        #Planning
        actions_taken, actions_planned, pred_recons, obs, try_obs = [], [], [], [], []  #(T), (?,3,D,D), (T,D,D,3) np, (T,D,D,3) np, (T,D,D,3)
        pred_recons = [initial_recon[0]]

        for t in range(num_actions_to_take):
            next_actions, goal_latent_index, pred_recon = action_selection_class.select_action(goal_info, cur_state_and_other_info["state"],
                                                                                               env, self.model, "{}".format(t))  # (Tp,A), (Sc), (3,D,D)

            try_obs.append(env.try_step(next_actions))  # (D,D,3)
            pred_recons.append(pred_recon)  # (3,D,D)

            next_obs = [env.get_observation()]
            for i in range(planning_horizon):
                next_obs.append(env.step(next_actions[i]))  # (D,D,3), np

            actions_taken.extend(next_actions[:planning_horizon])  # (Tt,A)
            actions_planned.append(next_actions)  # (Tp,A)
            obs.extend(next_obs[1:])  # Don't want to include starting image again

            # next_obs = np.array([env.step(an_action) for an_action in next_actions]) #(Tp=1,D,D,3) 255's numpy
            # actions.extend(next_actions)
            # obs.extend(next_obs)

            # self._remove_goal_latent(goal_info, goal_latent_index)
            next_obs = np.array(next_obs)  # (Tp+1,D,D,3) np
            cur_state_and_other_info = self.update_state(next_obs, next_actions[:planning_horizon],
                                cur_state_and_other_info["state"], file_name="{}/state_update_{}.png".format(self.logging_dir, t))


        ########Create final mpc image########
        # pdb.set_trace()
        goal_image_tensor = process_env_obs(goal_image)  # (1,3,D,D)
        starting_image_tensor = process_env_obs(initial_obs[-1])  # (1,3,D,D)

        obs = np.concatenate((initial_obs[-1:], obs))  # (T+1,D,D,3) np
        obs = process_env_obs(np.array(obs))  # (T+1,3,D,D)
        obs = torch.cat([obs, goal_image_tensor])  # (T+2,3,D,D)
        try_obs = process_env_obs(np.array(try_obs))  # (T,3,D,D)
        try_obs = torch.cat([starting_image_tensor, try_obs, goal_image_tensor])  # (T+2,3,D,D)
        pred_recons = torch.stack(pred_recons)  # (T+1,3,D,D)
        pred_recons = torch.cat([pred_recons, goal_info["final_recon"]])  # (T+2,3,D,D)
        save_image(torch.cat([obs, pred_recons, try_obs], dim=0), "{}/mpc.png".format(self.logging_dir), nrow=obs.shape[0])


        ########Compute result stats########
        final_obs = process_env_obs(env.get_observation())  # (1,3,D,D)
        torch_goal_image = process_env_obs(goal_image)  # (1,3,D,D)
        mse = ptu.get_numpy(torch.pow(final_obs - torch_goal_image, 2).mean())  # Compare final obs to goal obs (Sc), numpy

        correct = env.compute_accuracy(true_data)
        stats = {'mse': mse, 'correct': int(correct), 'actions': actions_taken}
        return stats


    def _filter_goal_image(self, goal_info, n_objects):
        goal_latents_mask = goal_info["masks"].squeeze(0).squeeze(1)  # (B=1,K,1,D,D) -> (K,1,D,D) -> (K,D,D)
        goal_latents_recon = goal_info["sub_images"].squeeze(0)  # (B=1,K,3,D,D) -> (K,3,D,D)

        vals, keep = torch.sort(goal_latents_mask.mean((1, 2)), descending=True)
        save_image(goal_latents_mask[keep].unsqueeze(1).repeat(1, 3, 1, 1), '{}/filter_goal_masks.png'.format(self.logging_dir))

        blank_image = process_env_obs(self.env._blank_observation)  # (1,3,D,D)
        blank_image = blank_image * goal_latents_mask.unsqueeze(1)  # (1,3,D,D)*(K,1,D,D) -> (K,3,D,D)

        save_image(blank_image, '{}/filter_blank_image.png'.format(self.logging_dir))
        save_image(goal_latents_recon, '{}/filter_blank_image_2.png'.format(self.logging_dir))

        difs = torch.abs(goal_latents_recon - blank_image)  # (K,3,D,D)
        difs = torch.where(difs > 10 / 255, difs, ptu.zeros_like(difs))  # .sum((1, 2)) #(K,3,D,D)
        difs = difs.sum(1)  # (K,D,D)
        save_image(difs.unsqueeze(1).repeat(1, 3, 1, 1), '{}/filter_difs.png'.format(self.logging_dir))
        difs = difs.sum((1, 2))  # (K)

        vals, keep = torch.sort(difs, descending=True)

        save_image(goal_latents_mask[keep].unsqueeze(1).repeat(1, 3, 1, 1), '{}/filter_goal_masks_sorted.png'.format(self.logging_dir))

        keep = keep[:n_objects]
        goal_info["state"]["post"]["samples"] = goal_info["state"]["post"]["samples"][:, keep]  # (B=1,K,R) -> (B=1,N_ob,R)
        goal_info["sub_images"] = goal_info["sub_images"][:, keep]  # (B=1,N_ob,3,D,D)
        goal_info["masks"] = goal_info["masks"][:, keep]  # (B,N_ob,1,D,D)
        goal_info["colors"] = goal_info["colors"][:, keep] # (B=1,N_ob,3,D,D)

        save_image(goal_info["sub_images"].squeeze(0), '{}/filter_mpc_goal_latents_recon.png'.format(self.logging_dir), nrow=10)
        return goal_info

    def _remove_goal_latent(self, goal_info, latent_index):
        def exclude_one(torch_array, index):
            if torch_array.shape[1] == 1:
                return None
            return torch.stack([torch_array[:, i] for i in range(torch_array.shape[1]) if i != index], dim=1)

        # pdb.set_trace()
        goal_info["colors"] = exclude_one(goal_info["colors"], latent_index) # (B=1,K,3,D,D)
        goal_info["masks"] = exclude_one(goal_info["masks"], latent_index)  # (B=1,K,1,D,D)
        goal_info["sub_images"] = exclude_one(goal_info["sub_images"], latent_index)  # (B=1,K,3,D,D)
        goal_info["state"]["post"]["samples"] = exclude_one(goal_info["state"]["post"]["samples"], latent_index)
        return goal_info


    #Input: numpy array goal_image (D,D,3)
    def goal_inference(self, goal_image, filter_goal_image):
        schedule = np.zeros(5) #5 refinement steps on goal image
        input_goal_image = process_env_obs(goal_image).unsqueeze(0) #(B=1,T=1,3,D,D)
        num_tries = 40
        input_goal_image = input_goal_image.repeat(num_tries, 1, 1, 1, 1)  # (Nt,T1,3,D,D)

        goal_info = self.model.batch_internal_inference(input_goal_image, None, None, schedule,
                                                        figure_path=self.logging_dir + "/goal_inference_general.png")
        #Note: goal_info["final_recon"] is (B,3,D,D)
        mses = torch.pow(goal_info["final_recon"] - input_goal_image.squeeze(1), 2).mean((1, 2, 3))
        sorted_mses, sorted_indices = torch.sort(mses, descending=False)
        best_index = sorted_indices[0]

        for akey in goal_info:
            if akey == "state":
                goal_info[akey] = self.model.select_specific_state(goal_info[akey], best_index)
            else:
                goal_info[akey] = goal_info[akey][best_index:best_index+1]

        goal_info["goal_image"] = process_env_obs(goal_image) #(B=1,3,D,D)

        if filter_goal_image:
            goal_info = self._filter_goal_image(goal_info, **filter_goal_image)
        else:
            visualizer.visualize_state_info(goal_info, file_name="{}/goal_inference_selected.png".format(self.logging_dir), true_image=input_goal_image[0])
        return goal_info

    #Inputs: obs (T1,D,D,3) np, actions (T1-1,A) or None np
    def state_acquisition(self, obs, actions):
        input_obs = process_env_obs(obs).unsqueeze(0) #(B=1,T1,3,D,D)
        num_tries = 40
        input_obs = input_obs.repeat(num_tries, 1, 1, 1, 1)  # (Nt,T1,3,D,D)

        if actions is not None:
            input_actions = process_env_actions(actions).unsqueeze(0) #(B=1,T1-1,A)
            input_actions = input_actions.repeat(num_tries, 1, 1)  # (Nt,T1,A)
        else:
            input_actions = None
        schedule = self.model.get_rprp_schedule(seed_steps=4, num_images=obs.shape[0], num_refine_per_physics=2)
        state_info = self.model.batch_internal_inference(input_obs, input_actions, None, schedule,
                                                         select_best=True, figure_path="{}/state_acquisition.png".format(self.logging_dir))
        # visualizer.visualize_state_info(state_info, file_name="{}/state_acquisition.png".format(self.logging_dir), true_image=input_obs[:, -1])
        return state_info


    #Input: next_obs (T1,D,D,3) np, next_actions (T1,A) np, cur_state
    #  Note: next_obs[0] is the initial image that cur_state should decode to (so we don't need to do initial seed steps on that)
    def update_state(self, next_obs, next_actions, cur_state, file_name=None):
        input_obs = process_env_obs(next_obs).unsqueeze(0)  # (B=1,T1,3,D,D)
        num_tries = 40
        input_obs = input_obs.repeat(num_tries, 1, 1, 1, 1)  # (Nt,T1,3,D,D)

        input_actions = process_env_actions(next_actions).unsqueeze(0)  # (B=1,T1,A)
        input_actions = input_actions.repeat(num_tries, 1, 1)  # (Nt,T1,A)

        schedule = self.model.get_rprp_schedule(seed_steps=0, num_images=next_obs.shape[0], num_refine_per_physics=3)
        state_info = self.model.batch_internal_inference(input_obs, input_actions, cur_state, schedule,
                                                         select_best=True, figure_path=file_name)
        # if file_name is not None:
        #     visualizer.visualize_state_info(state_info, file_name, true_image=input_obs[:, -1])
        return state_info




#########Loading data and running mpc#########
def copy_to_save_file(dir_str):
    base = get_module_path()
    shutil.copytree(base + '/rlkit/torch/iodine', dir_str+'/saved_torch_iodine_files')
    shutil.copy2(base + '/examples/mpc_v2/stage3/stage3_mpc.py', dir_str + '/saved_stage3_mpc.py')

def main(variant):
    from rlkit.core import logger
    copy_to_save_file(logger.get_snapshot_dir())
    seed = int(variant['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    module_path = get_module_path()

    ######Start goal info loading######
    # goal_idxs = range(n_goals)
    goal_idxs = range(variant["goal_start_end_range"][0], variant["goal_start_end_range"][1])
    stats = {'mse': [], 'correct': [], 'actions': []}
    ######End goal info loading######

    ######Start Model loading######
    op3_args = variant["op3_args"]
    op3_args['K'] = 4
    m = iodine_v2.create_model_v2(op3_args, op3_args['det_repsize'], op3_args['sto_repsize'], action_dim=4)

    # model_file = variant['model_file']
    model_file = module_path + '/examples/mpc_v2/stage3/saved_models/{}.pkl'.format(variant['model_file'])
    state_dict = torch.load(model_file)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    m.load_state_dict(new_state_dict)
    m.cuda()
    m.eval_mode = True
    ######End Model loading######

    goal_folder = module_path + '/examples/mpc/stage3/goals/objects_seed_{}/'.format(variant['number_goal_objects'])
    num_seed_frames = 1
    aggregate_stats = {}

    ######Start planning execution######
    for i, goal_idx in enumerate(goal_idxs):
        env = BlockPickAndPlaceEnv(num_objects=1, num_colors=None, img_dim=64,
                                   include_z=False)  # Note num_objects & num_colors do not matter

        ####Load goal and starting info
        with open(goal_folder + 'goal_data.pkl', 'rb') as f:
            goal_dict = pickle.load(f)
        goal_image = goal_dict["goal_image"][i]  # (D,D,3) np
        # frames = goal_dict["frames"][i]  # (T,D,D,3) np
        actions = goal_dict["actions"][i]  # (T-1,6) np
        seed_actions = env._post_process_actions(actions)  # (T-1,4) np
        goal_env_info = goal_dict["goal_env_info"][i]
        starting_env_info = goal_dict["starting_env_info"][i]

        #####Get seed steps
        env.set_env_info(starting_env_info)
        seed_frames = [env.get_observation()]
        if num_seed_frames > 1:
            seed_actions = seed_actions[:num_seed_frames]
            for an_action in seed_actions:
                seed_frames.append(env.step(an_action))
        else:
            seed_actions = None
        seed_frames = np.array(seed_frames)  # (T,D,D,3) np


        #####Set up mpc
        logging_directory = "{}/goal_{}".format(logger.get_snapshot_dir(), goal_idx)
        cost_class = Cost(logging_directory, **variant['cost_args'])
        cem_process = Stage3_CEM(logging_dir=logging_directory, score_actions_class=cost_class, **variant['cem_args'])
        mpc = Stage3_MPC(m, logging_directory)
        single_stats = mpc.run_plan(goal_image, env, seed_frames, seed_actions, cem_process, num_actions_to_take=1,
                                    planning_horizon=1, true_data=goal_env_info, filter_goal_image = False)
                                    # filter_goal_image={"n_objects": variant['number_goal_objects']})

        for k, v in single_stats.items():
            stats[k].append(v)

        with open(logger.get_snapshot_dir() + '/results_rolling.pkl', 'wb') as f:
            pickle.dump(stats, f)

        for k, v in stats.items():
            if k != 'actions':
                aggregate_stats[k] = float(np.mean(v))
        aggregate_stats["individual_correct"] = stats["correct"]
        aggregate_stats["num_goals_tried"] = i+1
        json.dump(aggregate_stats, open(logger.get_snapshot_dir() + '/results_stats.json', 'w'))



    with open(logger.get_snapshot_dir() + '/results_final.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # aggregate_stats = {}
    # for k,v in stats.items():
    #     if k != 'actions':
    #         aggregate_stats[k] = float(np.mean(v))
    # aggregate_stats["individual_correct"] = stats["correct"]
    # json.dump(aggregate_stats, open(logger.get_snapshot_dir() + '/results_stats.json', 'w'))
    ######End planning execution######






# CUDA_VISIBLE_DEVICES=1 python stage3_mpc.py -de 0 -s 0 -m [s64d64_v1_params,curriculum_aws_params,random_aws_params]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-de', '--debug', type=int, default=0)
    parser.add_argument('-s', '--split', type=int, required=True)
    parser.add_argument('-m', '--model_file', type=str, required=True)
    parser.add_argument('-mode', '--mode', type=str, default='here_no_doodad')
    parser.add_argument('-n', '--num_obs', type=str, default='here_no_doodad')
    args = parser.parse_args()

    num_splits = 1
    num_goals = 100
    start_idx = num_goals // num_splits * args.split
    end_idx = min(num_goals // num_splits * (args.split+1), num_goals)

    variant = dict(
        algorithm='MPC',
        op3_args=params_to_info[args.model_file]["op3_args"],
        cem_args=dict(
            cem_steps=3,
            num_samples=1000,
            time_horizon=1,
        ),
        cost_args=dict(
            core_type='final_recon',  # "subimage", "final_recon", "latent"
            compare_func='mse',    # mse, psuedo_intersect
            post_process='raw',
            aggregate='sum',
        ),
        goal_start_end_range=[start_idx, end_idx],
        debug=args.debug,
        model_file=args.model_file,
        number_goal_objects=1,
    )

    run_experiment(
        main,
        exp_prefix='iodine-mpc-stage3-n{}-{}-v2'.format(variant["number_goal_objects"], args.model_file),
        mode=args.mode,
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        region='us-west-2',
    )






