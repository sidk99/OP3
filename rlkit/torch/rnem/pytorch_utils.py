import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.metrics import adjusted_mutual_info_score
import pickle


#############################Start pytorch layer functions#############################
PYTORCH_FUNCTIONS = {
    'sigmoid': torch.sigmoid,
    'relu': F.relu,
    'elu': F.elu,
    'tanh': torch.tanh,
}

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def apply_act(x, act):
    x = PYTORCH_FUNCTIONS[act](x)
    return x

#x shape: (B, C, W, H)
def apply_LN_conv(x, ln_layer):
    #x = x.permute([0, 2, 3, 1]) #(B, W, H, C)
    x = ln_layer(x)
    #x = x.permute([0, 3, 1, 2]) #(B, C, W, H)
    return x


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.1)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.1)

def orthog_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal(m.weight)

def visualize_parameters(model, aString=None):
    if aString:
        print(aString)
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n, p.size(), p.data.norm(), "No grad")
        else:
            print(n, p.size(), p.data.norm(), p.grad.data.norm(), torch.max(p.grad.data))
#############################End pytorch layer functions#############################


#############################Start computational functions#############################
def custom_one_hot(batch, depth, device):
    the_size = batch.size()
    tmp = torch.eye(depth, device=device).index_select(0, batch.flatten().to(device))
    tmp = tmp.view([the_size[0], the_size[1], -1])
    return tmp.permute([0, 2, 1])

def np_softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return np.exp(e_x) / np.sum(np.exp(e_x), axis=axis, keepdims=True)
#############################End computational functions#############################


#############################Start pytorch gpu functions#############################
def set_device(gpu_id=-1):
    global device
    device = torch.device("cuda:" + str(gpu_id) if gpu_id > -1 else "cpu")
    # device = torch.device("cuda" if gpu_id > -1 else "cpu")

def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)
#############################End pytorch gpu functions#############################


#############################Start logging util functions#############################
def save_obj(obj, path_name):
    with open(path_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path_name):
    with open(path_name, 'rb') as f:
        return pickle.load(f)


def append_dictionaries(log_dict, dict):
    for key_word in dict:
        log_dict[key_word].append(dict[key_word])
    return log_dict


def log_file(stringText, fileName, shouldPrint=False):
    with open(fileName, "a") as myfile:
        myfile.write(stringText)
    if shouldPrint:
        print(stringText)
#############################End logging util functions#############################


#################Start ARI/AMI score functions##################
def evaluate_groups_seq(true_groups, predicted, weights):
    """ Compute the weighted AMI score and corresponding mean confidence for given gammas.
    :param true_groups: (T, B, 1, W, H, 1)
    :param predicted: (T, B, K, W, H, 1)
    :param weights: (T)
    :return: scores, confidences (B,)
    """
    w_scores, w_confidences = 0., 0.
    assert true_groups.ndim == predicted.ndim == 6, true_groups.shape

    for t in range(true_groups.shape[0]):
        scores, confidences = evaluate_groups(true_groups[t], predicted[t])

        w_scores += weights[t] * np.array(scores)
        w_confidences += weights[t] * np.array(confidences)

    norm = np.sum(weights)

    return w_scores/norm, w_confidences/norm


def evaluate_groups(true_groups, predicted):
    """ Compute the AMI score and corresponding mean confidence for given gammas.
    :param true_groups: (B, 1, W, H, 1)
    :param predicted: (B, K, W, H, 1)
    :return: scores, confidences (B,)
    """
    scores, confidences = [], []
    assert true_groups.ndim == predicted.ndim == 5, true_groups.shape
    batch_size, K = predicted.shape[:2]
    true_groups = true_groups.reshape(batch_size, -1)
    predicted = predicted.reshape(batch_size, K, -1)
    predicted_groups = predicted.argmax(1)
    predicted_conf = predicted.max(1)
    for i in range(batch_size):
        true_group = true_groups[i]
        idxs = np.where(true_group != 0.0)[0]
        scores.append(adjusted_mutual_info_score(true_group[idxs], predicted_groups[i, idxs]))
        confidences.append(np.mean(predicted_conf[i, idxs]))

    return scores, confidences


# Inputs:
#     groups: shape=(T, B, 1, W, H, 1)
#         These are the masks as stored in the hdf5 files
#     gammas: shape=(T, B, K, W, H, 1)
#         These are the gammas as predicted by the network
def pytorch_ari_score(groups, gammas, iter_weights, device):
    # print("groups: {}".format(groups.size()))
    # print("gammas: {}".format(gammas.size()))

    # ignore first iteration
    groups = groups[1:].contiguous()
    gammas = gammas[1:]
    # reshape gammas and convert to one-hot
    yshape = gammas.size()
    gammas = gammas.view([yshape[0] * yshape[1], yshape[2], yshape[3] * yshape[4] * yshape[5]])
    Y = custom_one_hot(torch.argmax(gammas, dim=1), yshape[2], device)

    # reshape masks
    gshape = groups.size()
    groups = groups.view([gshape[0] * gshape[1], 1, gshape[3] * gshape[4] * gshape[5]])
    G = custom_one_hot(groups[:, 0].type(torch.LongTensor), int(torch.max(groups).item())+1, device)
    #RV: Above one_hot values computed by using ground truth data, group ids are ints
    # now Y and G both have dim (B*T, K, N) where N=W*H*C

    # mask entries with group 0
    M = torch.clamp(groups, 0, 1).to(device, dtype=torch.float32) # RV: M is a binary mask of groups (Double check this!)
    n = torch.sum(M, dim=[1,2]).to(device, dtype=torch.float32) # RV: Sum the number of 1's
    DM = G * M  # RV: Masking G which is ground truth mask (this should not be needed)
    YM = Y * M  # RV: Masking gammas with ground truth mask (This looks slightly suspicious?!)
    # RV: Above makes sense for getting rid of background embedding layer but could lead to inaccurate ari scores
    #    as an embedding layer that guesses a larger area than the ground truth group will not be penalized

    # contingency table for overlap between G and Y
    nij = torch.einsum('bij,bkj->bki', [YM, DM])
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    # rand index
    rindex = torch.sum(nij * (nij - 1), dim=[1, 2], dtype=torch.float32)
    aindex = torch.sum(a * (a - 1), dim=1, dtype=torch.float32)
    bindex = torch.sum(b * (b - 1), dim=1, dtype=torch.float32)
    expected_rindex = aindex * bindex / (n * (n - 1) + 1e-6)
    max_rindex = (aindex + bindex) / 2

    ARI = (rindex - expected_rindex) / torch.clamp(max_rindex - expected_rindex, 1e-6, 1e6)
    ARI = ARI.view([yshape[0], yshape[1]])
    iter_weights = torch.from_numpy(iter_weights[:, None]).to(device, dtype=torch.float32)

    sum_iter_weights = torch.sum(iter_weights)
    seq_ARI = torch.mean(torch.sum(ARI * iter_weights, dim=0) / sum_iter_weights)
    last_ARI = torch.mean(ARI[-1])
    confidences = torch.sum(torch.max(gammas, dim=1, keepdim=True)[0] * M, dim=[1, 2]) / n
    confidences = confidences.view([yshape[0], yshape[1]])
    seq_conf = torch.mean(torch.sum(confidences * iter_weights, dim=0) / sum_iter_weights)
    last_conf = torch.mean(confidences[-1])
    return seq_ARI, last_ARI, seq_conf, last_conf
#################End ARI/AMI score functions##################

###########Start plotting code###########
def color_spines(ax, color, lw=2):
    for sn in ['top', 'bottom', 'left', 'right']:
        ax.spines[sn].set_linewidth(lw)
        ax.spines[sn].set_color(color)
        ax.spines[sn].set_visible(True)


def color_half_spines(ax, color1, color2, lw=2):
    for sn in ['top', 'left']:
        ax.spines[sn].set_linewidth(lw)
        ax.spines[sn].set_color(color1)
        ax.spines[sn].set_visible(True)

    for sn in ['bottom', 'right']:
        ax.spines[sn].set_linewidth(lw)
        ax.spines[sn].set_color(color2)
        ax.spines[sn].set_visible(True)


def get_gamma_colors(nr_colors):
    hsv_colors = np.ones((nr_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, nr_colors, endpoint=False) + 2/3) % 1.0
    color_conv = hsv_to_rgb(hsv_colors)
    return color_conv

def curve_plot(values_dict, coarse_range, fine_range):
    if fine_range is not None:
        fig, ax = plt.subplots(1, 2, figsize=(40, 10))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax = [ax]

    for key, values in values_dict.items():
        # coarse
        ax[0].plot(values, label=key)
        ax[0].set_xlabel('epochs')
        ax[0].axis([0, len(values), coarse_range[0], coarse_range[1]])
        ax[0].set_title("coarse range")
        ax[0].legend()

        # fine
        if fine_range is not None:
            ax[1].plot(values, label=key)
            ax[1].set_xlabel('epochs')
            ax[1].axis([0, len(values), fine_range[0], fine_range[1]])
            ax[1].set_title("fine range")
            ax[1].legend()

    return fig


def plot_stuff(values_dict, y_keys, save_prefix, y_range=None):
    x_axis = values_dict['epoch']
    for aKey in y_keys:
        if aKey not in values_dict:
            print("Incorrect key passed to pytorch_utils.plot_stuff: {}".format(aKey))
            continue
        fig = plt.figure()

        if len(x_axis) == len(values_dict[aKey]):
            plt.plot(x_axis, values_dict[aKey])
        else:
            plt.plot(values_dict[aKey])
        if y_range is not None:
            plt.ylim(y_range)

        fig.suptitle('{}'.format(aKey))
        fig.savefig(save_prefix + '{}.png'.format(aKey), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
###########End plotting code###########

#RV: Gamma shape - (F+1, 1, K, W, H, 1), inputs:  (T, 1, 1, W, H, 1), preds: (F+1, 1, K, W, H, 1), input_frame_hist = (F)
#Note: Gamma is F+1 as it includes the first initial hidden state
def overview_plot(i, gammas, preds, clean_img, corrupted=None, depth_vals=None, input_frame_hist=None, targ_frame_hist=None):
    # F = len(input_frame_hist)
    F, B, K, W, H, C = gammas.shape
    F -= 1  # the initialization doesn't count as iteration
    corrupted = corrupted if corrupted is not None else clean_img
    gamma_colors = get_gamma_colors(K+3) #Additional colors for rollout, physics, improvement

    # restrict to sample i and get rid of useless dims
    # inputs = inputs[:, i, 0]
    clean_img = clean_img[:, 0, 0, :, :] #RV: Now T, W, H, 1
    # gammas = gammas[:, i, :, :, :, 0]
    gammas = gammas[:, 0, :, :, :, 0] #RV: Now T, K, W, H
    if preds.shape[1] != B:
        preds = preds[:, 0]
    preds = preds[:, i] #RV: Now T, K, W, H, 1
    corrupted = corrupted[:, i, 0]

    clean_img = np.clip(clean_img, 0., 1.)
    preds = np.clip(preds, 0., 1.)
    corrupted = np.clip(corrupted, 0., 1.)

    # print("Gammas: {}".format(gammas.shape))
    # print("inputs: {}".format(inputs.shape))
    # print("preds: {}".format(preds.shape))
    # print("corrupted: {}".format(corrupted.shape))

    def plot_img(ax, data, cmap='Greys_r', xlabel=None, ylabel=None, border_color=None, scale=False, fixed_range=True):
        if scale is False:
            scale = 1
        else:
            scale = np.max(data) + 1e-6
            ax.set_xlabel("{:.3e}".format(scale))
        if data.shape[-1] == 1:
            if fixed_range:
                ax.matshow(data[:, :, 0]/scale, cmap=cmap, vmin=0., vmax=1., interpolation='nearest')
            else:
                ax.matshow(data[:, :, 0]/scale, cmap=cmap, interpolation='nearest')
        else:
            ax.imshow(data/scale, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(xlabel, color=border_color or 'k') if xlabel else None
        ax.set_ylabel(ylabel, color=border_color or 'k') if ylabel else None
        if border_color is not None:
            color_spines(ax, color=border_color)

    def plot_gamma(ax, gamma, xlabel=None, ylabel=None):
        # print("Plot gamma: {}".format(gamma.shape))
        gamma = np.transpose(gamma, [1, 2, 0])
        gamma = gamma.reshape(-1, gamma.shape[-1]).dot(gamma_colors[:K]).reshape(gamma.shape[:-1] + (3,))
        # print("Plot gamma: {}".format(gamma.shape))
        ax.imshow(gamma, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel) if xlabel else None
        ax.set_ylabel(ylabel) if ylabel else None

    nrows, ncols = (K + 4 + K, F + 1) #K+4 originally, +K for individual gammas
    if depth_vals is not None:
        depth_vals = depth_vals[:, 0, :, :, :, :]
        nrows, ncols = (K + 4 + 2*K + K, F + 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(2 * ncols, 2 * nrows))

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    plot_gamma(axes[2, 0], gammas[0], ylabel='Gammas')
    for k in range(K + 1):
        axes[k + 3, 0].set_visible(False)

    prev_target_ind = 0
    for step_at in range(1, F + 1):
        g = gammas[step_at] #K, W, H, 1
        p = preds[step_at]
        input_ind = input_frame_hist[step_at-1]
        target_ind = targ_frame_hist[step_at-1]

        # 0th row: Target image, color signifies physics vs improvement step
        if target_ind == prev_target_ind: #Improvement step
            plot_img(axes[0, step_at], clean_img[target_ind], border_color=gamma_colors[K])
        else:  #Physics step
            plot_img(axes[0, step_at], clean_img[target_ind], border_color=gamma_colors[K+1])
        axes[0, step_at].set_title("{}".format(target_ind))

        reconst = np.sum(g[:, :, :, None] * p, axis=0)
        plot_img(axes[1, step_at], reconst) #1st row: Predicted image
        plot_gamma(axes[2, step_at], g) #2nd row: Gammas
        for k in range(K): #Next k rows are individual mu's
            plot_img(axes[k + 3, step_at], p[k], border_color=tuple(gamma_colors[k]), ylabel=('mu_{}'.format(k) if step_at == 1 else None))

        #Plotting input image
        if input_ind == -1:
            dp = depth_vals[step_at-1] #Depth probs from previous step
            dp_probs = np_softmax(dp, 0)
            input_image = np.sum(dp_probs * preds[step_at-1], axis=0) #Predicted image from previous step
            plot_img(axes[K + 3, step_at], input_image, border_color=gamma_colors[K+2]) #Color means predicted input
        else:
            input_image = corrupted[input_ind]
            plot_img(axes[K + 3, step_at], input_image) #No color means ground truth input

        #Individual gamma plots
        for k in range(K):
            plot_img(axes[K + 4 + k, step_at], np.expand_dims(g[k], -1), border_color=tuple(gamma_colors[k]), ylabel=('gamma_{}'.format(k) if step_at == 1 else None))

        #Plotting depths and depth_probabilities
        if depth_vals is not None:
            dp = depth_vals[step_at]
            dp_probs = np_softmax(dp, 0)
            for k in range(K):
                plot_img(axes[K * 2 + 4 + k, step_at], dp[k], fixed_range=False)
                # plot_img(axes[K*2 + 4 + k, t], dp[k], scale=True)
                plot_img(axes[K*2 + 4 + k + K, step_at], dp_probs[k])
        prev_target_ind = target_ind

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig

#########Plotting one column at a time##########
def plot_columns(list_of_stuff, cur_column, ncols, cur_fig_axes=(None,None)):
    cur_fig, axes = cur_fig_axes[0], cur_fig_axes[1]
    if cur_fig is None:
        nrows = len(list_of_stuff)
        cur_fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

    for i in range(len(list_of_stuff)):
        tmp = list_of_stuff[i].view((64, 64, -1)).detach().cpu().numpy()
        # print(tmp.shape)
        if tmp.shape[-1] == 1:
            axes[i, cur_column].matshow(tmp[:, :, 0], cmap='Greys_r', interpolation='nearest')
        else:
            axes[i, cur_column].imshow(tmp, interpolation='nearest')
    return cur_fig, axes


import torch
import numpy as np


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


"""
GPU wrappers
"""

_use_gpu = False
device = None
_gpu_id = 0


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)
