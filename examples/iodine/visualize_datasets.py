import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu
# from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv
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

from examples.iodine.iodine import load_dataset


def load_model(variant, action_size):
    if variant['model'] == 'savp':
        time_horizon = variant['mpc_args']['time_horizon']
        m = SAVP_MODEL('/nfs/kun1/users/rishiv/Research/baseline/logs/pickplace_multienv_10k/ours_savp/', 'model-500000', 0,
                       batch_size=20, time_horizon=time_horizon)
    else:
        model_file = variant['model_file']

        if variant['model_type'] == 'next_step':
            variant['model']['refine_args']['added_fc_input_size'] = action_size
        elif variant['model_type'] == 'static':
            action_size = 0

        m = iodine.create_model(variant, action_dim=action_size)
        state_dict = torch.load(model_file)
        # pdb.set_trace()

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

#T is the total number of frames we want to evaluate on, so we do T-1 physics steps
#Frames is (bs, T1, ch, imsize, imsize),
def get_object_subimage_recon(frames, actions, model, model_type, T, image_type):
    def get_image_mse(frames, preds): #Both of size (bs, T, ch, D, D)
        return torch.pow(frames - preds, 2).mean((-1, -2, -3)) #(bs, T)

    seed_steps = 5
    if model_type == 'static':
        all_object_recons = []
        masks_recons = []
        for i in range(T):
            schedule = np.zeros(5) #Do 5 refine steps
            # pdb.set_trace()
            x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = model._forward_dynamic_actions(frames[:, i:i+1], actions, schedule)
            object_recons = x_hats * masks #(bs, 5, K, 3, D, D)
            object_recons = object_recons[:, -1] #(bs, K, 3, D, D)
            final_recons = object_recons.sum(1, keepdim=True)  # (bs, 1, 3, D, D)

            #If plotting masks
            # if image_type == 'masks':
            object_recons = masks.repeat(1, 1, 1, 3, 1, 1)  # (bs, 5, K, 3, D, D)
            object_recons = object_recons[:, -1]  # (bs, K, 3, D, D)

            #If plotting subimages with white background
            # masks = masks.repeat(1, 1, 1, 3, 1, 1)  # (bs, 5, K, 3, D, D)
            # masks = masks[:, -1]  # (bs, K, 3, D, D)
            # object_recons = torch.where(masks < 0.01, ptu.ones_like(object_recons), object_recons)

            tmp = torch.cat([final_recons, object_recons], dim=1) #(bs, K+1, 3, D, D)
            all_object_recons.append(tmp)
        all_object_recons = torch.stack(all_object_recons, dim=0) #(T, bs, K+1, 3, D, D))
        all_object_recons = all_object_recons.permute(1, 0, 2, 3, 4, 5).contiguous() #(bs, T, K+1, 3, D, D)
        mse = get_image_mse(frames[:, :T],  all_object_recons[:, :, 0]) #(bs, T)
        return all_object_recons, mse
    elif model_type == 'rprp':
        #T is the total number of frames, so we do T-1 physics steps
        num_refine_per_phys = 1
        schedule = np.zeros(seed_steps + (T-1)*num_refine_per_phys) #len(schedule) = T2
        schedule[seed_steps::num_refine_per_phys] = 1 #[0,0,0,0,1,0,1,0,1,0] if num_refine_per_phys=2 for example
        # pdb.set_trace()
        # frames is (bs, T, ch, imsize, imsize),
        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = model._forward_dynamic_actions(frames, actions, schedule)
        object_recons = x_hats * masks  # (bs, T2, K, 3, D, D)
        object_recons = object_recons[:, seed_steps-1::num_refine_per_phys] # (bs, T, K, 3, D, D)
        final_recons = object_recons.sum(2, keepdim=True) #(bs, T, 1, 3, D, D)

        # If plotting masks
        object_recons = masks.repeat(1, 1, 1, 3, 1, 1)  # (bs, T2, K, 3, D, D)
        object_recons = object_recons[:, seed_steps - 1::num_refine_per_phys]  # (bs, T, K, 3, D, D)

        # If plotting subimages with white background
        # masks = masks.repeat(1, 1, 1, 3, 1, 1)  # (bs, 5, K, 3, D, D)
        # masks = masks[:, seed_steps - 1::num_refine_per_phys]  # (bs, T, K, 3, D, D)
        # object_recons = torch.where(masks < 0.01, ptu.ones_like(object_recons), object_recons)

        all_object_recons = torch.cat([final_recons, object_recons], dim=2) #(bs, T, K+1, 3, D, D)
        mse = get_image_mse(frames[:, :T], all_object_recons[:, :, 0]) #(bs, T)

        return all_object_recons, mse
    elif model_type == 'rprp_pred':
        # T is the total number of frames, so we do T-1 physics steps
        num_refine_per_phys = 1
        schedule = np.zeros(seed_steps + (T - 1) * num_refine_per_phys)  # len(schedule) = T2
        schedule[seed_steps::num_refine_per_phys] = 1  # [0,0,0,0,1,0,1,0,1,0] if num_refine_per_phys=2 for example
        # pdb.set_trace()
        # frames is (bs, T, ch, imsize, imsize),
        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = model._forward_dynamic_actions(
            frames, actions, schedule)
        object_recons = x_hats * masks  # (bs, T2, K, 3, D, D)
        object_recons = object_recons[:, seed_steps::num_refine_per_phys]  # (bs, T-1, K, 3, D, D)
        final_recons = object_recons.sum(2, keepdim=True)  # (bs, T-1, 1, 3, D, D)

        # If plotting masks
        object_recons = masks.repeat(1, 1, 1, 3, 1, 1)  # (bs, T2, K, 3, D, D)
        object_recons = object_recons[:, seed_steps::num_refine_per_phys]  # (bs, T-1, K, 3, D, D)

        all_object_recons = torch.cat([final_recons, object_recons], dim=2)  # (bs, T-1, K+1, 3, D, D)
        padding = ptu.zeros([all_object_recons.shape[0], 1, *list(all_object_recons.shape[2:])])  # (bs, 1, K+1, 3, D, D)
        all_object_recons = torch.cat([padding, all_object_recons], dim=1)  # (bs, T, K+1, 3, D, D)
        mse = get_image_mse(frames[:, :T], all_object_recons[:, :, 0])  # (bs, T)
        return all_object_recons, mse
    elif model_type == 'next_step':
        schedule = np.ones(T-1) * 2
        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = model._forward_dynamic_actions(frames, actions, schedule)
        object_recons = x_hats * masks  # (bs, T-1, K, 3, D, D)
        final_recons = object_recons.sum(2, keepdim=True)  # (bs, T-1, 1, 3, D, D)

        # If plotting masks
        object_recons = masks.repeat(1, 1, 1, 3, 1, 1)  # (bs, T-1, K, 3, D, D)

        all_object_recons = torch.cat([final_recons, object_recons], dim=2)  # (bs, T-1, K+1, 3, D, D)
        padding = ptu.zeros([all_object_recons.shape[0], 1, *list(all_object_recons.shape[2:])]) #(bs, 1, K+1, 3, D, D)
        all_object_recons = torch.cat([padding, all_object_recons], dim=1) #(bs, T, K+1, 3, D, D)
        mse = get_image_mse(frames[:, :T], all_object_recons[:, :, 0]) #(bs, T)
        return all_object_recons, mse
    else:
        return ValueError("Invalid model_type: {}".format(model_type))



###################Start image creation functions###############
#High level: Want to run multiple different methods on the same frames and save that into a single image
def create_image(models_and_type, frames, actions, image_prefix, T, image_type):
    frames = frames.to(ptu.device)/255
    actions = actions.to(ptu.device)
    # frames is (bs, T1, ch, imsize, imsize)

    all_object_recons = []
    for model, model_type in models_and_type:
        object_recons, _ = get_object_subimage_recon(frames, actions, model, model_type, T, image_type) #(bs, T, K+1, 3, D, D)
        all_object_recons.append(object_recons)
    all_object_recons = torch.stack(all_object_recons, dim=0) #(M, bs, T, K, 3, imsize, imsize)

    all_object_recons = all_object_recons.permute(1, 2, 0, 3, 4, 5, 6).contiguous() #(bs, T, M, K, 3, D, D)
    cur_shape = all_object_recons.shape
    all_object_recons = all_object_recons.view(list(cur_shape[:2]) + [cur_shape[2]*cur_shape[3]] + list(cur_shape[4:])) #(bs, T, K*M, 3, D, D)

    for i in range(cur_shape[0]):
        tmp = frames[i, :cur_shape[1]].unsqueeze(1)  # (T, 1, 3, D, D)
        tmp = torch.cat([tmp, all_object_recons[i]], dim=1) #(T, 1, 3, D, D), (T, K*M, 3, D, D) -> (T, 1+K*M, 3, D, D)
        tmp = tmp.permute(1, 0, 2, 3, 4).contiguous()  #(1+K*M, T, 3, D, D)
        tmp = tmp.view(-1, *cur_shape[-3:])  # (T*(1+K*M), 3, D, D)
        save_image(tmp, filename=image_prefix+"_{}.png".format(i), nrow=cur_shape[1])

def create_images_from_dataset(variant):
    train_path = get_module_path() + '/ec2_data/{}.h5'.format(variant['dataset'])
    num_samples = 100
    train_dataset, _ = load_dataset(train_path, train=False, batchsize=1, size=num_samples, static=False)

    models_and_type = []
    for a_model in variant['models']:
        m = load_model(a_model, train_dataset.action_dim)
        m_type = a_model['model_type']
        models_and_type.append((m, m_type))

    image_indices = list(range(num_samples))
    for idx in image_indices:
        frames, actions = train_dataset[idx]
        frames = frames.unsqueeze(0)
        actions = actions.unsqueeze(0)
        create_image(models_and_type, frames, actions, logger.get_snapshot_dir()+"/image_{}".format(idx), variant['T'], variant['plot_type'])
###################End image creation functions###############

###################Start mse functions###############
def get_mse(models_and_type, frames, actions, T):
    frames = frames.to(ptu.device) / 255
    actions = actions.to(ptu.device)
    all_mse = []
    for model, model_type in models_and_type:
        all_object_recons, mse = get_object_subimage_recon(frames, actions, model, model_type, T, 'subimage') #mse is (bs, T)
        all_mse.append(mse)
    all_mse = torch.stack(all_mse, dim=0) #(M, bs, T)
    return all_mse

def get_mse_from_dataset(variant):
    train_path = get_module_path() + '/ec2_data/{}.h5'.format(variant['dataset'])
    num_samples = 100
    train_dataset, _ = load_dataset(train_path, train=False, batchsize=1, size=num_samples, static=False)

    models_and_type = []
    for a_model in variant['models']:
        m = load_model(a_model, train_dataset.action_dim)
        m_type = a_model['model_type']
        models_and_type.append((m, m_type))

    # image_indices = list(range(50))
    batch_indices = np.arange(0, num_samples, 4) #bs=4
    all_mse = []
    for i in range(len(batch_indices)-1):
        start_idx, end_idx = batch_indices[i], batch_indices[i+1]
        frames, actions = train_dataset[start_idx:end_idx] #(bs, T, 3, D, D)
        # pdb.set_trace()
        # frames = frames.unsqueeze(0)
        # actions = actions.unsqueeze(0)
        mse = get_mse(models_and_type, frames, actions, variant['T']) #(M, bs, T), torch tensors
        all_mse.append(mse.permute(1, 0, 2)) #(bs, M, T)
    all_mse = torch.stack(all_mse, dim=0) #(I/bs, bs, M, T)
    all_mse = ptu.get_numpy(all_mse.view(-1, len(models_and_type), variant['T'])) #(I, M, T), numpy array now
    np.save(logger.get_snapshot_dir() + '/computed_mse.npy', all_mse)

    mean_vals = np.mean(all_mse, axis=0) #(M, T)
    std_vals = np.std(all_mse, axis=0) #(M, T)
    for i in range(len(models_and_type)):
        if models_and_type[i][1] == 'next_step' or 'rprp_pred':
            plt.errorbar(range(1, variant['T']), mean_vals[i][1:], std_vals[i][1:], label='{}'.format(models_and_type[i][1]), capsize=5)
        else:
            plt.errorbar(range(0, variant['T']), mean_vals[i], std_vals[i], label='{}'.format(models_and_type[i][1]), capsize=5)

    # plt.legend(bbox_to_anchor=(0.4, 0.8), loc="upper right")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.yscale('log')
    plt.savefig(logger.get_snapshot_dir() + '/relative_mse.png', bbox_inches="tight")
###################End mse functions###############


###################Start analysis functions##############
def analyze_mse(variant):
    if variant['dataset'] == 'cloth':
        file_path = "/nfs/kun1/users/rishiv/Research/op3_exps/07-03-images-cloth/07-03-images-cloth_2019_07_03_01_40_45_0000--s-9047/computed_mse.npy"
    elif variant['dataset'] == 'poke':
        file_path = "/nfs/kun1/users/rishiv/Research/op3_exps/07-04-images-poke/07-04-images-poke_2019_07_04_00_09_54_0000--s-46658/computed_mse.npy"
    elif variant['dataset'] == 'solid':
        file_path = '/nfs/kun1/users/rishiv/Research/op3_exps/07-06-images-solid/07-06-images-solid_2019_07_06_05_41_24_0000--s-13353/computed_mse.npy'
    else:
        raise ValueError("Invalid dataset given")

    all_mse = np.load(file_path) #(I, M, T)
    #Models are rprp, static, next_step in that order
    T = all_mse.shape[-1]
    rprp_vs_static = all_mse[:,1] - all_mse[:,0] #(I, T)
    rprp_vs_next_step = all_mse[:, 2] - all_mse[:,0] #(I, T)

    print("RPRP: mean={}, std={}. \nStatic: mean={}. std={}".format(np.mean(all_mse[:,0], axis=0), np.std(all_mse[:,0], axis=0),
                                                                    np.mean(all_mse[:,1], axis=0), np.mean(all_mse[:,1], axis=0)))


    plt.figure()
    plt.errorbar(range(T), np.mean(rprp_vs_static, axis=0), np.std(rprp_vs_static, axis=0), label='rprp vs static', capsize=5)
    plt.savefig(logger.get_snapshot_dir() + '/rprp_vs_static.png')

    plt.figure()
    plt.errorbar(range(1, T), np.mean(rprp_vs_next_step, axis=0)[1:], np.std(rprp_vs_next_step, axis=0)[1:], label='rprp vs next_step',capsize=5)
    plt.savefig(logger.get_snapshot_dir() + '/rprp_vs_next_step.png')
###################End analysis functions##############


#################Create graphs#############
def create_mse_graphs(variant):
    if variant['dataset'] == 'cloth':
        file_path = "/nfs/kun1/users/rishiv/Research/op3_exps/07-06-images-cloth/07-06-images-cloth_2019_07_06_06_15_12_0000--s-93491/computed_mse.npy"
    elif variant['dataset'] == 'poke':
        file_path = "/nfs/kun1/users/rishiv/Research/op3_exps/07-06-images-poke/07-06-images-poke_2019_07_06_05_29_35_0000--s-93758/computed_mse.npy"
    elif variant['dataset'] == 'solid':
        file_path = '/nfs/kun1/users/rishiv/Research/op3_exps/07-06-images-solid/07-06-images-solid_2019_07_06_05_41_24_0000--s-13353/computed_mse.npy'
    else:
        raise ValueError("Invalid dataset given")

    all_mse = np.load(file_path)  # (I, M, T)
    # Models are rprp, rprp_pred, static, next_step in that order

    T = all_mse.shape[-1] #
    static_mse = all_mse[:, 2] #(I, T)
    next_step_mse = all_mse[:, 3] #(I, T)
    rprp = all_mse[:, 0] #(I, T)
    rprpr_next_step = all_mse[:, 1] #(I, T)

    plt.figure()
    plt.errorbar(range(T), np.mean(static_mse, axis=0), np.std(static_mse, axis=0), label='Static Iodine', capsize=5, color='blue', linestyle='--')
    plt.errorbar(range(T), np.mean(rprp, axis=0), np.std(rprp, axis=0), label='OP3', capsize=5, color='green', linestyle='--')

    # plt.errorbar(range(1, T), np.mean(next_step_mse, axis=0)[1:], np.std(static_mse, axis=0)[1:], label='Sequence', capsize=5, color='blue', linestyle='-')
    # plt.errorbar(range(1, T), np.mean(rprpr_next_step, axis=0)[1:], np.std(rprpr_next_step, axis=0)[1:], label='OP3_pred', capsize=5, color='green', linestyle='-')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(logger.get_snapshot_dir() + '/comparison.png', bbox_inches="tight")

#Cloth models:
# models = [dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
#                                model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_54_10_0000--s-775/_params.pkl",
#                                model_type="rprp"),
#                   dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_54_10_0000--s-775/_params.pkl",
#                        model_type="rprp_pred"),
#                           dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
#                                model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-iodine-blocks-cloth-static-iodine/06-28-iodine-blocks-cloth-static-iodine_2019_06_28_06_04_51_0000--s-29681/_params.pkl",
#                                model_type="static"),
#                           dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
#                                model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-29-cloth-next-step/06-29-cloth-next_step_2019_06_29_05_34_47_0000--s-95282/_params.pkl",
#                                model_type="next_step")],

#Poke models:
# models = [dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-poke-rprp-reg/07-04-poke-rprp-reg_2019_07_04_23_58_16_0000--s-38337/_params.pkl",
#                        model_type="rprp"),
#                   dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-poke-rprp-reg/07-04-poke-rprp-reg_2019_07_04_23_58_16_0000--s-38337/_params.pkl",
#                        model_type="rprp_pred"),
#                   dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-iodine-blocks-poke-static-iodine/06-28-iodine-blocks-poke-static-iodine_2019_06_28_05_55_05_0000--s-63777/_params.pkl",
#                        model_type="static"),
#                   dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-29-poke-next-step/06-29-poke-next_step_2019_06_29_05_36_03_0000--s-98425/_params.pkl",
#                        model_type="next_step")],

#Solid models:
# models = [dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_46_09_0000--s-59196/_params.pkl",
#                        model_type="rprp"),
#                   dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_46_09_0000--s-59196/_params.pkl",
#                        model_type="rprp_pred"),
#                   dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-30-solid-static-iodine/06-30-solid-static_iodine_2019_06_30_12_55_14_0000--s-12264/_params.pkl",
#                        model_type="static"),
#                   dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
#                        model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-01-solid-next-step/07-01-solid-next_step_2019_07_01_08_40_56_0000--s-50539/_params.pkl",
#                        model_type="next_step")],
#/nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_54_10_0000--s-775
# /nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_46_09_0000--s-59196

dataset_to_models = dict(
    cloth = [dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                               model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_46_09_0000--s-59196/_params.pkl",
                               model_type="rprp"),
                  dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-cloth-rprp-reg/07-04-cloth-rprp-reg_2019_07_04_23_46_09_0000--s-59196/_params.pkl",
                       model_type="rprp_pred"),
                          dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                               model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-iodine-blocks-cloth-static-iodine/06-28-iodine-blocks-cloth-static-iodine_2019_06_28_06_04_51_0000--s-29681/_params.pkl",
                               model_type="static"),
                          dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                               model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-29-cloth-next-step/06-29-cloth-next_step_2019_06_29_05_34_47_0000--s-95282/_params.pkl",
                               model_type="next_step")],
    poke = [dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-poke-rprp-reg/07-04-poke-rprp-reg_2019_07_04_23_58_16_0000--s-38337/_params.pkl",
                       model_type="rprp"),
                  # dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
                  #      model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-poke-rprp-reg/07-04-poke-rprp-reg_2019_07_04_23_58_16_0000--s-38337/_params.pkl",
                  #      model_type="rprp_pred"),
                  dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-iodine-blocks-poke-static-iodine/06-28-iodine-blocks-poke-static-iodine_2019_06_28_05_55_05_0000--s-63777/_params.pkl",
                       model_type="static")],
                  # dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
                  #      model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-29-poke-next-step/06-29-poke-next_step_2019_06_29_05_36_03_0000--s-98425/_params.pkl",
                  #      model_type="next_step")],
    solid = [dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-06-solid-rprp-reg/07-06-solid-rprp-reg_2019_07_06_08_20_29_0000--s-48759/_params.pkl",
                       model_type="rprp"),
                  # dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                  #      model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-06-solid-rprp-reg/07-06-solid-rprp-reg_2019_07_06_08_20_29_0000--s-48759/_params.pkl",
                  #      model_type="rprp_pred"),
                  dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-30-solid-static-iodine/06-30-solid-static_iodine_2019_06_30_12_55_14_0000--s-12264/_params.pkl",
                       model_type="static")],
                  # dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=5,
                  #      model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-01-solid-next-step/07-01-solid-next_step_2019_07_01_08_40_56_0000--s-50539/_params.pkl",
                  #      model_type="next_step")],
)

#Example usage: CUDA_VISIBLE_DEVICES=0 python visualize_datasets.py -da cloth
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-da', '--dataset', type=str, default=None, required=True) # stack_o2p2_60k, pickplace_1env_1k
    parser.add_argument('-m', '--mode', type=str,default='here_no_doodad')

    args = parser.parse_args()

    #models: list of dictionaries containing the information of the models to be loaded/run
    #Images will be of form: T images wide, 1st row is true images, then next rows are dictated by models, where
    #  each model takes K+1 rows, where the 1st row is the combined reconstruction and the next K are the subimages
    variant = dict(
        models=dataset_to_models[args.dataset],
        # models=[dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
        #              model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-poke-rprp-reg/07-04-poke-rprp-reg_2019_07_04_23_58_16_0000--s-38337/_params.pkl",
        #              model_type="rprp"),
        #         dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
        #              model_file="/nfs/kun1/users/rishiv/Research/op3_exps/07-04-poke-rprp-reg/07-04-poke-rprp-reg_2019_07_04_23_58_16_0000--s-38337/_params.pkl",
        #              model_type="rprp_pred"),
        #         dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
        #              model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-iodine-blocks-poke-static-iodine/06-28-iodine-blocks-poke-static-iodine_2019_06_28_05_55_05_0000--s-63777/_params.pkl",
        #              model_type="static"),
        #         dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
        #              model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-29-poke-next-step/06-29-poke-next_step_2019_06_29_05_36_03_0000--s-98425/_params.pkl",
        #              model_type="next_step")],
        # models=[dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
        #              model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-cloth-rprp/06-28-cloth-rprp_2019_06_28_06_49_47_0000--s-74526/_params.pkl",
        #              model_type="rprp"),
        #         dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
        #              model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-iodine-blocks-cloth-static-iodine/06-28-iodine-blocks-cloth-static-iodine_2019_06_28_06_04_51_0000--s-29681/_params.pkl",
        #              model_type="static")],
        #         # dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
        #         #      model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-29-poke-next-step/06-29-poke-next_step_2019_06_29_05_36_03_0000--s-98425/_params.pkl",
        #         #      model_type="next_step")],
        plot_type="masks", #masks or subimage
        T=8,
        dataset=args.dataset,
        machine_type='g3.16xlarge' #Ignore: Only a logging tool and NOT used for setting ec2 instances (do that in conf)
    )

    #Relevant options: 'here_no_doodad', 'local_docker', 'ec2'
    run_experiment(
        create_images_from_dataset, #get_mse_from_dataset, create_images_from_dataset, analyze_mse, create_mse_graphs
        exp_prefix='images-{}'.format(args.dataset),
        mode=args.mode,
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=None,
        region='us-west-2'
    )




