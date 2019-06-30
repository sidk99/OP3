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

#T is the total number of frames, so we do T-1 physics steps
#Frames is (bs, T1, ch, imsize, imsize),
def get_object_masks_recon(frames, actions, model, model_type, T):
    seed_steps = 5
    if model_type == 'static':
        all_object_recons = []
        for i in range(T):
            schedule = np.zeros(5) #Do 5 refine steps
            # pdb.set_trace()
            x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = model._forward_dynamic_actions(frames[:, i:i+1], actions, schedule)
            object_recons = x_hats * masks #(bs, 5, K, 3, D, D)
            object_recons = object_recons[:, -1] #(bs, K, 3, D, D)
            final_recons = object_recons.sum(1, keepdim=True)  # (bs, 1, 3, D, D)
            tmp = torch.cat([final_recons, object_recons], dim=1) #(bs, K+1, 3, D, D)
            all_object_recons.append(tmp)
        all_object_recons = torch.stack(all_object_recons, dim=0) #(T, bs, K+1, 3, D, D))
        all_object_recons = all_object_recons.permute(1, 0, 2, 3, 4, 5).contiguous()
        return all_object_recons
    elif model_type == 'rprp':
        #T is the total number of frames, so we do T-1 physics steps
        schedule = np.zeros(seed_steps + (T-1)*2) #len(schedule) = T2
        schedule[seed_steps::2] = 1 #[0,0,0,0,1,0,1,0,1,0]
        # pdb.set_trace()
        # frames is (bs, T, ch, imsize, imsize),
        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = model._forward_dynamic_actions(frames, actions, schedule)
        object_recons = x_hats * masks  # (bs, T2, K, 3, D, D)
        object_recons = object_recons[:, seed_steps-1::2] # (bs, T, K, 3, D, D)
        final_recons = object_recons.sum(2, keepdim=True) #(bs, T, 1, 3, D, D)
        all_object_recons = torch.cat([final_recons, object_recons], dim=2) #(bs, T, K+1, 3, D, D)
        return all_object_recons
    elif model_type == 'next_step':
        schedule = np.ones(T-1) * 2
        x_hats, masks, total_loss, kle_loss, log_likelihood, mse, final_recon, lambdas = model._forward_dynamic_actions(frames, actions, schedule)
        object_recons = x_hats * masks  # (bs, T-1, K, 3, D, D)
        final_recons = object_recons.sum(2, keepdim=True)  # (bs, T-1, 1, 3, D, D)
        all_object_recons = torch.cat([final_recons, object_recons], dim=2)  # (bs, T-1, K+1, 3, D, D)
        padding = ptu.zeros([all_object_recons.shape[0], 1, *list(all_object_recons.shape[2:])]) #(bs, 1, K+1, 3, D, D)
        all_object_recons = torch.cat([padding, all_object_recons], dim=1) #(bs, T, K+1, 3, D, D)
        return all_object_recons
    else:
        return ValueError("Invalid model_type: {}".format(model_type))



#High level: Want to run multiple different methods on the same frames and save that into a single image
def create_image(models_and_type, frames, actions, image_prefix, T):
    frames = frames.to(ptu.device)/255
    actions = actions.to(ptu.device)
    # frames is (bs, T1, ch, imsize, imsize)

    all_object_recons = []
    for model, model_type in models_and_type:
        object_recons = get_object_masks_recon(frames, actions, model, model_type, T) #(bs, T, K+1, 3, D, D)
        all_object_recons.append(object_recons)
    all_object_recons = torch.stack(all_object_recons, dim=0) #(M, bs, T, K, 3, imsize, imsize)

    all_object_recons = all_object_recons.permute(1, 2, 0, 3, 4, 5, 6).contiguous() #(bs, T, M, K, 3, D, D)
    cur_shape = all_object_recons.shape
    all_object_recons = all_object_recons.view(list(cur_shape[:2]) + [cur_shape[2]*cur_shape[3]] + list(cur_shape[4:])) #(bs, T, K*M, 3, D, D)

    for i in range(cur_shape[0]):
        tmp = frames[i, :cur_shape[1]].unsqueeze(1)  # (T, 1, 3, D, D)
        tmp = torch.cat([tmp, all_object_recons[i]], dim=1) #(T, 1, 3, D, D), (T, K*M, 3, D, D) -> (T, 1+K*M, 3, D, D)
        tmp = tmp.permute(1, 0, 2, 3, 4).contiguous()  #(T, 1+K*M, 3, D, D)
        tmp = tmp.view(-1, *cur_shape[-3:])  # (T*(1+K*M), 3, D, D)
        save_image(tmp, filename=image_prefix+"_{}.png".format(i), nrow=cur_shape[1])


def create_multiple_images(variant):
    train_path = get_module_path() + '/ec2_data/{}.h5'.format(variant['dataset'])
    train_dataset, T = load_dataset(train_path, train=True, batchsize=1, size=100, static=False)

    models_and_type = []
    for a_model in variant['models']:
        m = load_model(a_model, train_dataset.action_dim)
        m_type = a_model['model_type']
        models_and_type.append((m, m_type))

    image_indices = list(range(4))
    for idx in image_indices:
        frames, actions = train_dataset[idx]
        frames = frames.unsqueeze(0)
        actions = actions.unsqueeze(0)
        create_image(models_and_type, frames, actions, logger.get_snapshot_dir()+"/image_{}".format(idx), variant['T'])



#Example usage: CUDA_VISIBLE_DEVICES=4,5 python visualize_datasets.py -da cloth
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-da', '--dataset', type=str, default=None, required=True) # stack_o2p2_60k, pickplace_1env_1k
    parser.add_argument('-m', '--mode', type=str,default='here_no_doodad')

    args = parser.parse_args()

    #models: list of dictionaries containing the information of the models to be loaded/run
    #Images will be of form: T images wide, 1st row is true images, then next rows are dictated by models, where
    #  each model takes K+1 rows, with the 1st row is the combined reconstruction and the next K are the subimages
    variant = dict(
        models = [dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-cloth-rprp/06-28-cloth-rprp_2019_06_28_06_49_47_0000--s-74526/_params.pkl",
                       model_type="rprp"),
                  dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-28-iodine-blocks-cloth-static-iodine/06-28-iodine-blocks-cloth-static-iodine_2019_06_28_06_04_51_0000--s-29681/_params.pkl",
                       model_type="static"),
                  dict(model=iodine.imsize64_large_iodine_architecture_multistep_physics, K=4,
                       model_file="/nfs/kun1/users/rishiv/Research/op3_exps/06-29-cloth-next-step/06-29-cloth-next_step_2019_06_29_05_34_47_0000--s-95282/_params.pkl",
                       model_type="next_step")],
        T=4,
        dataset=args.dataset,
        machine_type='g3.16xlarge' #Ignore: Only a logging tool and NOT used for setting ec2 instances (do that in conf)
    )

    #Relevant options: 'here_no_doodad', 'local_docker', 'ec2'
    run_experiment(
        create_multiple_images,
        exp_prefix='images-{}'.format(args.dataset),
        mode=args.mode,
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
        seed=None,
        region='us-west-2'
    )




