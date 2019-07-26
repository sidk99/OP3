from torchvision.utils import save_image
import torch

#images (W,H,3,D,D), values should be between 0 and 1
def create_image_from_subimages(images, file_name):
    cur_shape = images.shape
    images = images.permute(1, 0, 2, 3, 4).contiguous()  # (H,W,3,D,D)
    images = images.view(-1, *cur_shape[-3:])  # (H*W, 3, D, D)
    save_image(images, filename=file_name, nrow=cur_shape[1])

#true_images (T,3,D,D),  colors (T,K,3,D,D),  masks (T,K,1,D,D),  file_name (string)
#quicksave_type is either "full" or "subimages"
def quicksave(true_images, colors, masks, file_name, quicksave_type):
    recons = (colors * masks).sum(1) #(T,3,D,D)
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



