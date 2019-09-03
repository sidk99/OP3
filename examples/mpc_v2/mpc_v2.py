from rlkit.torch import pytorch_util as ptu
from rlkit.torch.iodine.visualizer import quicksave
from rlkit.util.plot import plot_multi_image
import torch
import numpy as np
# from abc import ABC, abstractmethod

##############Cost Class ############
class Cost:
    def __init__(self, logging_directory, latent_or_subimage='subimage', compare_func='mse', post_process='raw', aggregate='sum'):
        self.remove_goal_latents = False
        self.logging_directory = logging_directory
        self.latent_or_subimage = latent_or_subimage
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
                             '{}/cost_{}.png'.format(self.logging_directory, self.image_suffix), caption=caption)
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
                             '{}/mpc_pred_{}.png'.format(self.logging_directory, self.image_suffix), caption=caption)
        return ptu.get_numpy(sorted_costs), ptu.get_numpy(best_action_idxs), ptu.get_numpy(min_goal_latent_idx)

    def mse(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

    def image_mse(self, im1, im2):
        # im1, im2 are (*, 3, D, D)
        # Note: * dimensions may not be equal between im1, im2 so automatically broadcast over them
        return torch.pow(im1 - im2, 2).mean((-1, -2, -3)) #Takes means across the last dimensions (3, D, D)


#############Action Selection Class#########
####Stage1 Specific
class Stage1_CEM:
    def __init__(self, cem_steps, num_samples, time_horizon, goal_info, env, score_actions_class):
        self.cem_steps = cem_steps
        self.num_samples = num_samples
        self.time_horizon = time_horizon
        self.goal_info = goal_info
        self.env = env
        self.score_actions_class = score_actions_class

    def select_action(self, h_cur, model):
        return self._cem(model)

    # Inputs: action_images (B,3,D,D),  model
    # Outputs: Index of actions based off sorted costs (B)
    def _random_shooting(self, action_images, model):
        goal_info = self.goal_info
        schedule = np.array([0, 0, 0, 0, 1])
        predicted_info = model.batched_internal_inference(obs=action_images, actions=None, initial_hidden_state=None,
                                                          schedule=schedule, figure_path=None)
        sorted_costs, best_actions_indices, goal_latent_indices = self.score_actions_class.get_action_rankings(
            goal_info["state"]["samples"], goal_info["final_recon"], goal_info["goal_image"],
            predicted_info["state"]["samples"], predicted_info["sub_images"], predicted_info["final_recon"])
        return best_actions_indices

    # Inputs: N/A
    # Outputs: actions (B,T,A),  action_images (B,3,D,D)
    def _get_initial_actions(self):
        actions = []
        for i in range(self.time_horizon):
            actions.append(np.stack([self.env.sample_action() for _ in range(self.num_samples)]))
        action_images = ptu.from_numpy(np.moveaxis(np.stack([self.env.try_action(action) for action in actions]), 3, 1))/255
        return actions, action_images

    # Input: model
    # Output: Best action (T,A)
    def _cem(self, model):
        actions, action_images = self._get_initial_actions()
        filter_cutoff = int(self.num_samples * 0.1) #F

        for i in range(self.cem_steps):
            best_actions_indices = self._random_shooting(action_images, model) #(B)
            sorted_actions = actions[best_actions_indices] #(B,T,A)
            best_actions = sorted_actions[:filter_cutoff] #(F,T,A)
            mean = best_actions.mean(0) #(T,A)
            std = best_actions.std(0) #(T,A)
            actions = self.env.sample_multiple_action_gaussian(mean, std, self.num_samples)
        return actions[0]





##############MPC Class ############
class MPC:
    def __init__(self, num_actions_to_plan):
        self.tmp = -1
        self.num_actions_to_plan = num_actions_to_plan

    # Inputs: goal_image (3,D,D), env, initial_obs (T1,3,D,D), initial_actions (T1,A), action_selection_class, T
    # Assume all images are between 0 & 1 and are tensors
    def run_plan(self, goal_image, env, initial_obs, initial_actions, model, action_selection_class, T):
        #Goal inference
        goal_state_and_other_info = self.goal_inference(goal_image)

        #State acquisition
        cur_state_and_other_info = self.state_acquisition(initial_obs, initial_actions)

        #Planning
        for t in range(T):
            next_actions = action_selection_class.select_action(cur_state_and_other_info, goal_state_and_other_info)
            next_obs = np.array([env.step(an_action) for an_action in next_actions]) #(Tp,D,D,3) 255's numpy
            cur_state_and_other_info = self.update_state(next_obs, next_actions, cur_state_and_other_info["state"])

    #Input: env_obs (D,D,3) or (T,D,D,3), values between 0-255, numpy array
    def process_env_obs(self, env_obs):
        if len(env_obs.shape) == 3: #(D,D,3) numpy
            env_obs = np.expand_dims(env_obs, 0) #(T=1,D,D,3), numpy
        return ptu.from_numpy(np.moveaxis(env_obs, 3, 1))/255

    # Input: env_obs (A) or (T,A), numpy array
    def process_env_actions(self, env_actions):
        if len(env_actions.shape) == 1: #(A) numpy
            env_actions = np.expand_dims(env_actions, 0) #(T=1,A), numpy
        return ptu.from_numpy(env_actions)

    #Input: numpy array goal_image (D,D,3)
    def goal_inference(self, goal_image):
        schedule = np.zeros(5) #5 refinement steps on goal image
        input_goal_image = self.process_env_obs(goal_image) #(1,3,D,D)
        return self.internal_inference(input_goal_image, None, None, schedule)

    #Inputs: obs (T1,D,D,3) numpy, actions (T1-1,A) numpy
    def state_acquisition(self, obs, actions):
        input_obs = self.process_env_obs(obs) #(T1,3,D,D)
        input_actions = self.process_env_actions(actions) #(T1-1,A)
        schedule = self.m.get_rprp_schedule(seed_steps=5, num_images=obs.shape[0], num_refine_per_physics=2)
        return self.internal_inference(input_obs, input_actions, None, schedule)

    #Input: next_obs (T1,D,D,3) numpy, next_actions (T1,A) numpy, cur_state
    def update_state(self, next_obs, next_actions, cur_state):
        input_obs = self.process_env_obs(next_obs)  # (T1,3,D,D)
        input_actions = self.process_env_actions(next_actions)  # (T1,A)
        schedule = self.m.get_rprp_schedule(seed_steps=2, num_images=next_obs.shape[0], num_refine_per_physics=2)
        return self.internal_inference(input_obs, input_actions, cur_state, schedule)

    # Input: next_obs (T1,3,D,D) or None, next_actions (T2,A) or None, initial_hidden_state, schedule (T3)
    # Assume inputs are all pytorch tensors in proper format
    # Output: Note the values are all pytorch values!
    def internal_inference(self, obs, actions, initial_hidden_state, schedule, figure_path=None):
        loss_schedule = np.zeros_like(schedule)

        #Output: colors (T1,B,K,3,D,D), masks (T1,B,K,1,D,D), final_recon (B,3,D,D),
        # total_loss, total_kle_loss, total_clog_prob, mse are all (Sc), end_hidden_state
        self.m.eval_mode = False
        colors, masks, final_recon, total_loss, total_kle_loss, total_clog_prob, mse, cur_hidden_state = \
            self.m.run_schedule(obs, actions, initial_hidden_state, schedule=schedule, loss_schedule=loss_schedule)

        cur_hidden_state = self.m.detach_state_info(cur_hidden_state)
        important_values = {
            "colors": colors[-1], #(B,K,3,D,D)
            "masks": masks[-1],  #(B,K,1,D,D)
            "sub_images": colors[-1] * masks[-1], #(B,K,3,D,D)
            "final_recon": final_recon, #(B,3,D,D)
            "state": cur_hidden_state
        }
        if figure_path is not None:
            quicksave(obs, colors, masks, schedule, figure_path, "full")
        return important_values


