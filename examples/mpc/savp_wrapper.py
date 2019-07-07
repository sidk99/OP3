import argparse
import errno
import json
import os
import random

import numpy as np
import torch
import tensorflow as tf
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import rlkit.torch.pytorch_util as ptu
# from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
from video_prediction import datasets, models


class SAVP_MODEL:
    def __init__(self, check_point_path, checkpoint_version, gpu_mem_frac, batch_size, time_horizon):
        model_hparams_dict = {}

        # Loading in parameters from the checkpoint
        checkpoint_dir = check_point_path
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            # print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            model_type = options['model']
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")

        VideoPredictionModel = models.get_model_class(model_type)
        hparams_dict = dict(model_hparams_dict)
        hparams_dict.update({
            'context_frames': 1,
            'sequence_length': time_horizon+1,
            'repeat': 1
        })

        model = VideoPredictionModel(mode='test', hparams_dict=hparams_dict)

        # sequence_length = model.hparams.sequence_length
        # context_frames = model.hparams.context_frames
        # future_length = sequence_length - context_frames

        # image_shape = (batch_size, 2, 64, 64, 3) #(bs, T=2, w, h, 3)
        # action_shape = (batch_size, 1, 13) #(bs, T-1=1, 13)
        image_shape = (batch_size, time_horizon+1, 64, 64, 3)  # (bs, T=2, w, h, 3)
        action_shape = (batch_size, time_horizon, 4)  # (bs, T-1=1, 13)

        # pdb.set_trace()

        self.input_phs = {'images': tf.placeholder(tf.float32, image_shape, 'images_ph'),
                          'actions': tf.placeholder(tf.float32, action_shape, 'actions_ph')}

        with tf.variable_scope(''):
            model.build_graph(self.input_phs)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.graph.as_default()

        model.restore(self.sess, check_point_path + checkpoint_version)
        self.model = model

        self.fake_latent_size = 5
        self.time_horizon = time_horizon
        self.K = 1


    #Input: obs is a pytorch tensor (B, 3, D, D)
    #Output: pred_obs (B, 3, D, D), obs_latents (B, K, rep_size), obs_latents_recon (B, K, 3, D, D)
    def step(self, obs, action):
        bs = obs.shape[0]
        obs = np.transpose(ptu.get_numpy(obs), (0, 2, 3, 1))# * 255  # (B, D, D, 3)
        # obs = (obs).astype(np.uint8)
        obs = np.expand_dims(obs, 1) #(B, 1, D, D, 3)
        # obs = np.repeat(obs, self.time_horizon, axis=1) #(B, T, D, D, 3)
        obs = np.tile(obs, (1, self.time_horizon+1, 1, 1, 1)) #(B, T, D, D, 3)

        action = ptu.get_numpy(action[:, :, (0, 1, 3, 4)])
        feed_dict = {self.input_phs['images'] : obs, self.input_phs['actions']: action}
        images = self.sess.run(self.model.outputs['gen_images'], feed_dict=feed_dict) #(B, T, D, D, 3)
        # pdb.set_trace()
        pred_obs = images[:, -1]
        pred_obs = ptu.from_numpy(np.transpose(pred_obs, (0, 3, 1, 2))) #/255 #(B, 3, D, D)
        obs_latents = ptu.randn((bs, 1, self.fake_latent_size)) #(B, K=1, rep_size)
        obs_latents_recon = pred_obs.unsqueeze(1) #(B, K=1, 3, D, D)
        return pred_obs, obs_latents, obs_latents_recon

    # Batch method for step
    def step_batched(self, inputs, actions, bs=4):
        # Handle large obs in batches
        n_batches = int(np.ceil(inputs.shape[0] / float(bs)))
        outputs = [[], [], []]

        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, inputs.shape[0])
            if actions is not None:
                actions_batch = actions[start_idx:end_idx]
            else:
                actions_batch = None

            pred_obs, obs_latents, obs_latents_recon = self.step(inputs[start_idx:end_idx], actions_batch)
            outputs[0].append(pred_obs)
            outputs[1].append(obs_latents)
            outputs[2].append(obs_latents_recon)

        return torch.cat(outputs[0]), torch.cat(outputs[1]), torch.cat(outputs[2])

    #Input: goal_image_tensor: (1, 3, D, D)
    #Output: (3, D, D), (K, rep_size), (K, 3, D, D), (K, D, D)
    def refine(self, goal_image_tensor, hidden_state, plot_latents):
        imsize = goal_image_tensor.shape[-1]

        recon = goal_image_tensor.view(3, imsize, imsize)
        latents = ptu.randn((1, self.fake_latent_size))
        sub_images = goal_image_tensor
        masks = ptu.ones((1, imsize, imsize))
        return recon, latents, sub_images, masks


