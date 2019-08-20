import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
import shutil
# from rlkit.launchers.ray.launcher import launch_experiment
from torch.distributions import Normal
import pickle
import torch
import torch.nn as nn
from argparse import ArgumentParser
import imageio
from rlkit.envs.blocks.mujoco.block_stacking_env import BlockEnv
from rlkit.core import logger
from rlkit.core.timer import timer
from torchvision.utils import save_image
from rlkit.util.plot import plot_multi_image
import json
import os
import rlkit.torch.iodine.iodine as iodine
from collections import OrderedDict
from rlkit.util.misc import get_module_path
# from ray import tune
import time
import pdb
print("Hello Sid")

class Cost:
    def __init__(self, type, logger_prefix_dir):
        self.type = type
        self.remove_goal_latents = True
        self.logger_prefix_dir = logger_prefix_dir

    def best_action(self, mpc_step, goal_latents, goal_latents_recon, goal_image, pred_latents,
                    pred_latents_recon, pred_image, actions, cem_step):
        if self.type == 'min_min_latent':
            return self.min_min_latent(mpc_step, goal_latents, goal_latents_recon, goal_image,
                                       pred_latents, pred_latents_recon, pred_image, actions)
        if self.type == 'sum_goal_min_latent':
            self.remove_goal_latents = False
            return self.sum_goal_min_latent(mpc_step, goal_latents, goal_latents_recon, goal_image,
                                            pred_latents, pred_latents_recon, pred_image, actions)

        elif self.type == 'goal_pixel':
            return self.goal_pixel(goal_latents, goal_image, pred_latents, pred_image, actions)
        elif self.type == 'latent_pixel':
            return self.latent_pixel(mpc_step, goal_latents_recon, goal_image, pred_latents_recon,
                                     pred_image, actions, cem_step)
        else:
            raise Exception

    def mse(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

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
        best_action_idxs = costs.min(0)[0].sort()[1]
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

        return best_pred_obs, actions[ptu.get_numpy(best_action_idxs)], best_goal_idx

    def sum_goal_min_latent(self, mpc_step, goal_latents, goal_latents_recon, goal_image,
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

        best_action_idxs = costs.sum(0).sort()[1]
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

        # plot_multi_image(ptu.get_numpy(full_plot),
        #                  logger.get_snapshot_dir() + '%s/mpc_pred_%d.png' % (
        #                      self.logger_prefix_dir, mpc_step),
        #                  caption=caption)

        return best_pred_obs, actions[ptu.get_numpy(best_action_idxs)], best_goal_idx

    def goal_pixel(self, goal_latents, goal_image, pred_latents, pred_image, actions):
        mse = torch.pow(pred_image - goal_image, 2).mean(3).mean(2).mean(1)

        min_cost, action_idx = mse.min(0)

        return ptu.get_numpy(pred_image[action_idx]), actions[action_idx], 0

    def latent_pixel(self, mpc_step, goal_latents_recon, goal_image, pred_latents_recon, pred_image,
                     actions, cem_step):
        # goal_latents_recon (5, 3, 64, 64)
        # goal_image (1, 3, 64, 64)
        # pred_latents_recon (960, K, 3, 64, 64)
        # pred_image (960, 3, 64, 64)
        # actions (960, 13)

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
            #cost: (960, K)
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

        costs = torch.stack(costs)  # (n_goal_latents, n_actions)
        latent_idxs = torch.stack(latent_idxs)  # (n_goal_latents, n_actions)
        best_action_idxs = costs.min(0)[0].sort()[1] #Sorted indices of best actions (n_actions)
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx])  # Gets single predicted image corresponding with best action

        matching_costs, matching_goal_idx = costs.min(0) #Each is (n_actions)

        matching_latent_idx = latent_idxs[matching_goal_idx, np.arange(n_actions)] #Grabs the predicted latent index that matches the closest goal image

        matching_goal_rec = torch.stack([goal_latents_recon[j] for j in matching_goal_idx]) #(960, 3, 64, 64)
        matching_latent_rec = torch.stack([pred_latents_recon[i][matching_latent_idx[i]] for i in range(n_actions)])  #(960, 3, 64, 64)
        best_pred_obs = ptu.get_numpy(pred_image[best_action_idx]) #Gets single predicted image corresponding with best action

        full_plot = torch.cat([pred_image.unsqueeze(0), #(1, 960, 3, 64, 64)
                               pred_latents_recon.permute(1, 0, 2, 3, 4), #(K, 960, 3, 64, 64)
                               matching_latent_rec.unsqueeze(0), #(1, 960, 3, 64, 64)
                               matching_goal_rec.unsqueeze(0)], 0) #(1, 960, 3, 64, 64)
        #full_plot of shape (K+3, 960, 3, 64, 64)
        # pdb.set_trace()

        plot_size = 8
        full_plot = full_plot[:, ptu.get_numpy(best_action_idxs[:plot_size])] #Selects 8 best actions (K+3, 960, 3, 64, 64)
        caption = np.zeros(full_plot.shape[:2]) #(K+3, 960)
        caption[1:1 + K, :] = ptu.get_numpy(torch.stack(costs_latent).min(0)[0].permute(1, 0))[:,:plot_size]
        caption[-2, :] = matching_costs.cpu().numpy()[ptu.get_numpy(best_action_idxs[:plot_size])]

        plot_multi_image(ptu.get_numpy(full_plot),
                         logger.get_snapshot_dir() + '%s/mpc_pred_%d_%d.png' % (
                             self.logger_prefix_dir, mpc_step, cem_step),
                         caption=caption)
        # self.plot_image(costs, costs_latent, mpc_step, pred_image, pred_latents_recon, matching_latent_rec, matching_goal_rec)


        return best_pred_obs, actions[ptu.get_numpy(best_action_idxs)], best_goal_idx


    # def plot_image(self, costs, costs_latent, mpc_step, pred_image, pred_latents_recon, matching_latent_rec, matching_goal_rec):
    #     #costs: (n_goal_latents, n_actions)
    #     K = pred_latents_recon.shape[1]
    #
    #     matching_costs, matching_goal_idx = costs.min(0)  # Each is (n_actions)
    #     best_action_idxs = costs.min(0)[0].sort()[1]
    #
    #     full_plot = torch.cat([pred_image.unsqueeze(0),  # (1, 960, 3, 64, 64)
    #                            pred_latents_recon.permute(1, 0, 2, 3, 4),  # (K, 960, 3, 64, 64)
    #                            matching_latent_rec.unsqueeze(0),  # (1, 960, 3, 64, 64)
    #                            matching_goal_rec.unsqueeze(0)], 0)  # (1, 960, 3, 64, 64)
    #     # full_plot of shape (K+3, 960, 3, 64, 64)
    #
    #     plot_size = 8
    #     full_plot = full_plot[:, ptu.get_numpy(best_action_idxs[:plot_size])]
    #     caption = np.zeros(full_plot.shape[:2])
    #     caption[1:1 + K, :] = ptu.get_numpy(torch.stack(costs_latent).min(0)[0].permute(1, 0))[:,:plot_size]
    #     caption[-2, :] = matching_costs.cpu().numpy()[ptu.get_numpy(best_action_idxs[:plot_size])]
    #
    #     plot_multi_image(ptu.get_numpy(full_plot),
    #                      logger.get_snapshot_dir() + '%s/mpc_pred_%d.png' % (self.logger_prefix_dir, mpc_step),
    #                      caption=caption)





class MPC:
    def __init__(self, model, env, n_actions, mpc_steps,
                 n_goal_objs=3,
                 cost_type='latent_pixel',
                 filter_goals=False,
                 true_actions=None,
                 logger_prefix_dir=None,
                 mpc_style="random_shooting",  # options are random_shooting, cem
                 cem_steps=2,
                 use_action_image=True,  # True for stage 1, False for stage 3
                 true_data=None,
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
            if not os.path.exists(logger.get_snapshot_dir() + logger_prefix_dir):
                os.mkdir(logger.get_snapshot_dir() + logger_prefix_dir)

        self.logger_prefix_dir = logger_prefix_dir
        self.use_action_image = use_action_image
        self.true_data = true_data  # ground truth target

    def filter_goal_latents(self, goal_latents, goal_latents_mask, goal_latents_recon):
        # Keep top goal latents with highest mask area except first
        n_goals = self.n_goal_objs
        vals, keep = torch.sort(goal_latents_mask.mean(2).mean(1), descending=True)
        # goal_latents_recon[keep[n_goals]] += goal_latents_recon[keep[n_goals + 1]]
        keep = keep[1:1 + n_goals]
        goal_latents = torch.stack([goal_latents[i] for i in keep])
        goal_latents_recon = torch.stack([goal_latents_recon[i] for i in keep])

        save_image(goal_latents_recon,
                   logger.get_snapshot_dir() + '%s/mpc_goal_latents_recon.png' %
                   self.logger_prefix_dir, nrow=10)

        return goal_latents, goal_latents_recon

    def remove_idx(self, array, idx):
        return torch.stack([array[i] for i in set(range(array.shape[0])) - set([idx])])

    def run(self, goal_image):
        goal_image_tensor = ptu.from_numpy(np.moveaxis(goal_image, 2, 0)).unsqueeze(
            0).float()[:, :3] / 255.  # (1, 3, imsize, imsize)

        rec_goal_image, goal_latents, goal_latents_recon, goal_latents_mask = self.model.refine(
            goal_image_tensor,
            hidden_state=None,
            plot_latents=False)  # (K, rep_size)

        # Keep top 4 goal latents with greatest mask area excluding 1st (background)
        if self.filter_goals:
            goal_latents, goal_latents_recon = self.filter_goal_latents(goal_latents,
                                                                        goal_latents_mask,
                                                                        goal_latents_recon)

        obs = self.env.reset()/255
        print(np.max(obs), np.mean(obs))

        obs_lst = [np.moveaxis(goal_image.astype(np.float32) / 255., 2, 0)[:3]]
        pred_obs_lst = [ptu.get_numpy(rec_goal_image)]
        try_action_obs_list = [ptu.get_numpy(rec_goal_image)] #RV

        actions = []
        for mpc_step in range(self.mpc_steps):
            t0 = time.time()
            pred_obs, action, goal_idx = self.step_mpc(obs, goal_latents, goal_image_tensor,
                                                       mpc_step,
                                                       goal_latents_recon)
            mpc_time = time.time()
            actions.append(action)
            try_action_obs_list.append(np.moveaxis(self.env.try_action(action)/255, 2, 0))
            obs = self.env.step(action)/255
            step_time = time.time()
            pred_obs_lst.append(pred_obs)
            obs_lst.append(np.moveaxis(obs, 2, 0))
            if goal_latents.shape[0] == 1:
                break
            # remove matching goal latent from goal latents
            if self.cost.remove_goal_latents:
                goal_latents = self.remove_idx(goal_latents, goal_idx)
                goal_latents_recon = self.remove_idx(goal_latents_recon, goal_idx)
            #print(mpc_time - t0, step_time - mpc_time)
        # save_image(ptu.from_numpy(np.stack(obs_lst + pred_obs_lst)),
        #            logger.get_snapshot_dir() + '%s/mpc.png' % self.logger_prefix_dir,
        #            nrow=len(obs_lst))
        save_image(ptu.from_numpy(np.stack(obs_lst + pred_obs_lst + try_action_obs_list)),
                   logger.get_snapshot_dir() + '%s/mpc.png' % self.logger_prefix_dir,
                   nrow=len(obs_lst))
        # pdb.set_trace()

        # Compare final obs to goal obs
        mse = np.square(ptu.get_numpy(goal_image_tensor.squeeze().permute(1, 2, 0)) - obs).mean()
        # pdb.set_trace()
        (correct, max_pos, max_rgb), state = self.env.compute_accuracy(self.true_data)
        np.save(logger.get_snapshot_dir() + '%s/block_pos.p' % self.logger_prefix_dir, state)
        stats = {'mse': mse, 'correct': int(correct), 'max_pos': max_pos, 'max_rgb': max_rgb}
        return stats, np.stack(actions)

    def model_step_batched(self, obs, actions, bs=8):

        # Handle large obs in batches
        n_batches = int(np.ceil(obs.shape[0] / float(bs)))
        outputs = [[], [], []]

        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, obs.shape[0])
            actions_batch = actions[start_idx:end_idx].contiguous() if not self.use_action_image else None
            pred_obs, obs_latents, obs_latents_recon = self.model.step(obs[start_idx:end_idx].contiguous(), actions_batch)
            outputs[0].append(pred_obs)
            outputs[1].append(obs_latents)
            outputs[2].append(obs_latents_recon)

        return torch.cat(outputs[0]), torch.cat(outputs[1]), torch.cat(outputs[2])

    def step_mpc(self, obs, goal_latents, goal_image, mpc_step, goal_latents_recon):
        if self.mpc_style == 'random_shooting':
            best_pred_obs, best_actions, best_goal_idx = self._random_shooting_step(obs,
                                                                                    goal_latents,
                                                                                    goal_image,
                                                                                    mpc_step,
                                                                                    goal_latents_recon)
            return best_pred_obs, best_actions[0], best_goal_idx
        elif self.mpc_style == 'cem':
            return self._cem_step(obs, goal_latents, goal_image, mpc_step, goal_latents_recon)

    def _cem_step(self, obs, goal_latents, goal_image, mpc_step, goal_latents_recon):

        actions = None
        filter_idx = int(self.n_actions * 0.1)
        print("NEW ACTION")
        for i in range(self.cem_steps):
            best_pred_obs, best_actions, best_goal_idx = self._random_shooting_step(obs,
                                                                                    goal_latents,
                                                                                    goal_image,
                                                                                    mpc_step,
                                                                                    i,
                                                                                    goal_latents_recon,
                                                                                    actions=actions)
            best_actions = best_actions[:filter_idx]
            mean = best_actions.mean(0)
            std = best_actions.std(0)
            actions = np.stack([self.env.sample_action_gaussian(mean, std) for _ in range(self.n_actions)])

            mse = np.square(ptu.get_numpy(goal_image.squeeze()) - best_pred_obs).mean()
            print(mse)

        return best_pred_obs, best_actions[0], best_goal_idx

    def _random_shooting_step(self, obs, goal_latents, goal_image, mpc_step,
                              cem_step, goal_latents_recon,
                              actions=None):

        # obs is (imsize, imsize, 3)
        # goal latents is (<K, rep_size)

        if actions is None:
            actions = np.stack([self.env.sample_action() for _ in range(self.n_actions)])

        # polygox_idx, pos, axangle, rgb
        if self.true_actions is not None:
            actions = np.concatenate([self.true_actions[mpc_step].reshape((1, -1)), actions])



        if self.use_action_image:
            obs_rep = ptu.from_numpy(np.moveaxis(np.stack([self.env.try_action(action) for action in actions]), 3, 1))/255
            # obs_rep = ptu.from_numpy(np.moveaxis(np.stack(self.env.try_actions(actions)), 3, 1))/255
            # tmp = ptu.get_numpy(obs_rep)
            # print(np.max(tmp), np.min(tmp), np.max(tmp)*255, np.min(tmp)*255)
        else:
            obs_rep = ptu.from_numpy(np.moveaxis(obs, 2, 0)).unsqueeze(0).repeat(actions.shape[0],
                                                                                 1, 1,
                                                                                 1)


        pred_obs, obs_latents, obs_latents_recon = self.model_step_batched(obs_rep, ptu.from_numpy(actions))
        # pdb.set_trace()
        #goal_latents (5,128)
        #goal_latents_recon (5, 3, 64, 64)
        #goal_image (1, 3, 64, 64)
        #obs_latents (960, K, 128)
        #obs_latents_recon (960, K, 3, 64, 64)
        #pred_obs (960, 3, 64, 64)
        #actions (960, 13)


        best_pred_obs, best_actions, best_goal_idx = self.cost.best_action(mpc_step, goal_latents,
                                                                           goal_latents_recon,
                                                                           goal_image,
                                                                           obs_latents,
                                                                           obs_latents_recon,
                                                                           pred_obs,
                                                                           actions,
                                                                           cem_step)

        return best_pred_obs, best_actions, best_goal_idx


def copy_to_save_file():
    dir_str = logger.get_snapshot_dir()
    base = get_module_path()
    shutil.copy2(base+'/rlkit/torch/iodine/iodine.py', dir_str)
    shutil.copy2(base+'/rlkit/torch/iodine/iodine_trainer.py', dir_str)
    shutil.copy2(base+'/rlkit/torch/iodine/physics_network.py', dir_str)
    shutil.copy2(base+'/rlkit/torch/iodine/refinement_network.py', dir_str)
    shutil.copy2(base+'/examples/mpc/stage1/mpc_stage1.py', dir_str)

def main(variant):
    copy_to_save_file()
    # module_path = os.path.expanduser('~') + '/objects/rlkit'
    # model_file = module_path + '/saved_models/iodine-blocks-stack50k/SequentialRayExperiment_0_2019-05-15_01-24-38zy_wn4_6/model_params.pkl'
    # module_path = os.path.expanduser('~') + '/Research/fun_rlkit'
    module_path = '/nfs/kun1/users/rishiv/Research/fun_rlkit'
    # model_file = module_path + '/examples/mpc/saved_models/iodine_params_5_15.pkl'
    # model_file = '/nfs/kun1/users/rishiv/Research/op3_exps/05-29-iodine-blocks-stack-o2p2-60k/05-29-iodine-blocks-stack_o2p2_60k_2019_05_29_00_55_37_0000--s-81417/params.pkl'
    # model_file = '/nfs/kun1/users/rishiv/Research/op3_exps/05-29-iodine-blocks-stack-o2p2-60k/05-29-iodine-blocks-stack_o2p2_60k_2019_05_29_23_06_32_0000--s-93500/params.pkl'

    model_file = variant['model_file'] #+ '/_params.pkl'

    # model_file = module_path + \
    #              '/saved_models/iodine-blocks-stack_o2p2_60k/SequentialRayExperiment_0_2019' \
    #              '-05' \
    #              '-20_16-24-523j6_e94i/model_params.pkl'
    # if variant['structure'][1] > 7:
    #     # variant['model']['vae_kwargs']['K'] = variant['structure'][1] + 2
    #     variant['model']['K'] = variant['structure'][1] + 2
    variant['K'] = variant['structure'][1] + 2
    # pdb.set_trace()
    m = iodine.create_model(variant, 0)
    state_dict = torch.load(model_file)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    m.load_state_dict(new_state_dict)
    m.cuda()
    # m = nn.DataParallel(m)

    m.set_eval_mode(True)
    # pdb.set_trace()

    n_goals = 1 if variant['debug'] == 1 else 10
    goal_idxs = range(n_goals)

    # module_path = os.path.expanduser('~') + '/objects/rlkit'

    actions_lst = []
    stats = {'mse': 0, 'correct': 0, 'max_pos': 0, 'max_rgb': 0}
    goal_counter = 0
    structure, n_goal_obs = variant['structure']
    #n_goal_obs = self.variant['n_goal_obs']


    for i, goal_idx in enumerate(goal_idxs):
        goal_file = module_path + '/examples/mpc/stage1/manual_constructions/%s/%d_1.png' % (structure, goal_idx)
        # goal_file = module_path + '/examples/mpc/stage1/goals_3/img_%d.png' % goal_idx
        true_data = np.load(
            module_path + '/examples/mpc/stage1/manual_constructions/%s/%d.p' % (structure, goal_idx), allow_pickle=True)
        # pdb.set_trace()
        env = BlockEnv(n_goal_obs)
        mpc = MPC(m, env, n_actions=500, mpc_steps=n_goal_obs, true_actions=None,
                  cost_type=variant['cost_type'], filter_goals=True, n_goal_objs=n_goal_obs,
                  logger_prefix_dir='/%s_goal_%d' % (structure, goal_idx),
                  mpc_style=variant['mpc_style'], cem_steps=3, use_action_image=True,
                  true_data=true_data)
        goal_image = imageio.imread(goal_file)
        single_stats, actions = mpc.run(goal_image)
        for k, v in single_stats.items():
            stats[k] += v
        actions_lst.append(actions)
        goal_counter += 1
        np.save(logger.get_snapshot_dir() + '/optimal_actions.npy', np.stack(actions_lst))

    for k, v in stats.items():
        stats[k] /= float(goal_counter)
    print(stats)
    json.dump(stats, open(logger.get_snapshot_dir() + '/stats.json', 'w'))
    # np.save(logger.get_snapshot_dir() + '/optimal_actions.npy', np.stack(actions_lst))

#CUDA_VISIBLE_DEVICES=7 python mpc_stage1.py -de 1 -s
#CUDA_VISIBLE_DEVICES=0 python mpc_stage1.py -de 0 -s 1 -m /nfs/kun1/users/rishiv/Research/op3_exps/08-17-stack-o2p2-60k-single-step-physics-reg/08-17-stack_o2p2_60k-single_step_physics-reg-k5_2019_08_17_12_24_19_0000--s-36145/_params.pkl


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-de', '--debug', type=int, default=0)
    parser.add_argument('-s', '--split', type=int, required=True)
    parser.add_argument('-m', '--model_file', type=str, required=True)

    args = parser.parse_args()


    structures = [
        ('bridge', 5), #1
        ('pyramid', 6), #1
        ('pyramid-triangle', 6), #2
        ('spike', 6), #2
        ('stacked', 5), #3
        ('tall-bridge', 7), #3
        ('three-shapes', 5), #4
        ('double-bridge', 8),  #4
        ('double-bridge-close', 8),  #5
        ('double-bridge-close-topfar', 8),  #5
        ('towers', 9), #6
    ]
    n = 3 #(11+1)//n is the number of splits
    splits = [structures[i:i + n] for i in range(0, len(structures), n)]
    structure_split = splits[args.split]
    # pdb.set_trace()


    for s in structure_split:
        variant = dict(
            algorithm='MPC',
            cost_type='latent_pixel',  # 'sum_goal_min_latent' 'latent_pixel
            mpc_style='cem',  # random_shooting or cem
            model=iodine.imsize64_large_iodine_architecture_multistep_physics,
            structure=s,
            debug=args.debug,
            model_file=args.model_file,
            # n_goal_obs=structures[s_idx][1]
        )

        n_seeds = 1
        exp_prefix = 'iodine-mpc-stage1-k5-final'

        run_experiment(
            main,
            exp_prefix=exp_prefix,
            mode='here_no_doodad',
            variant=variant,
            use_gpu=True,  # Turn on if you have a GPU
        )

