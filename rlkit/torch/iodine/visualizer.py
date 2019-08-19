import numpy as np
import torch
from torchvision.utils import save_image
from rlkit.torch import pytorch_util as ptu
import pdb

#images (W,H,3,D,D), values should be between 0 and 1
def create_image_from_subimages(images, file_name):
    cur_shape = images.shape
    images = images.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
    images = images.view(-1, *cur_shape[-3:])  # (H*W, 3, D, D)
    save_image(images, filename=file_name, nrow=cur_shape[0])
    #Note: Weird that it is cur_shape[0] but cur_shape[1] produces incorrect image

#true_images (T1,3,D,D),  colors (T,K,3,D,D),  masks (T,K,1,D,D), schedule (T)
# file_name (string),  quicksave_type is either "full" or "subimages"
#Images are torch tensors, schedule is numpy array
def quicksave(true_images, colors, masks, schedule, file_name, quicksave_type):
    recons = (colors * masks).sum(1) #(T,3,D,D)

    #If we are doing rollouts (i.e. T1 >= sum(schedule))
    true_images = torch.cat([true_images, torch.zeros_like(true_images[:1].to(true_images.device))], dim=0)
    tmp = np.where(np.cumsum(schedule) < true_images.shape[0] - 1, np.cumsum(schedule), -1)
    true_images = true_images[tmp]
    # true_images = true_images[min(np.cumsum(schedule), true_images.shape[0]-1)] #(T,3,D,D) #NOTE: This only works for schedules with 0's and 1's!!

    # true_images = torch.where(np.cumsum(schedule) < true_images.shape[0], true_images, ptu.zeros_like(true_images))

    full_plot = torch.cat([true_images.unsqueeze(1), recons.unsqueeze(1)], dim=1) #(T,2,3,D,D)
    if quicksave_type == "full":
        subimages = colors * masks #(T,K,3,D,D)
        masks = masks.repeat(1,1,3,1,1) #(T,K,3,D,D)
        full_plot = torch.cat([full_plot, masks, subimages], dim=1) #(T,2+K+K,3,D,D)
    elif quicksave_type == "subimages":
        subimages = colors * masks #(T,K,3,D,D)
        full_plot = torch.cat([full_plot, subimages], dim=1)  # (T,2+K,3,D,D)
    else:
        raise ValueError("Invalid value '{}' given to quicksave".format(quicksave_type))
    create_image_from_subimages(full_plot, file_name)



