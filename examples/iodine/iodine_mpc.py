from torch import nn

from rlkit.torch.vae.iodine import IodineVAE

from rlkit.torch.conv_networks import BroadcastCNN
import rlkit.torch.vae.iodine as iodine
from rlkit.torch.vae.refinement_network import RefinementNetwork
from rlkit.torch.vae.physics_network import PhysicsNetwork
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
import rlkit.torch.vae.conv_vae as conv_vae
from rlkit.torch.vae.unet import UNet
from rlkit.torch.vae.iodine_trainer import IodineTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.pythonplusplus import identity
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.core import logger
import numpy as np
from scipy import misc
import h5py
import pickle
import scipy.misc
import mujoco_py as mjc
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from o2p2.mujoco.XML import XML
from o2p2.mujoco.logger import Logger

def load_dataset(data_path, train=True):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features'])
    elif 'TwoBall' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features'])
        else:
            feats = np.array(hdf5_file['test']['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        return data
    elif 'BlocksGeneration' in data_path:
        if train:
            feats = np.array(hdf5_file['training']['features'])
        else:
            feats = np.array(hdf5_file['validation']['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        data = np.swapaxes(data, 2, 3)
        return data


def execute_actions(actions, polygons, asset_path, sim_steps=2000, img_dim=64,
                    hold_last_action=False):
    polygon_map = {ind: ply for ind, ply in enumerate(polygons)}

    xml = XML(asset_path)
    names = []
    for action in actions:
        ply_ind, pos, axangle, scale, rgb = action
        ply = polygon_map[ply_ind]
        rgba = rgb.tolist() + [1]
        #print('Dropping {} | pos: {} | axangle: {} | scale: {} | rgb: {} '.format(ply, pos, axangle, scale, rgb))

        name = xml.add_mesh(ply, pos=pos, axangle=axangle, scale=scale, rgba=rgba)
        names.append(name)

    xml_str = xml.instantiate()
    model = mjc.load_model_from_xml(xml_str)
    sim = mjc.MjSim(model)

    log_steps = len(actions) + 1
    logger = Logger(xml, sim, steps=log_steps, img_dim=img_dim)
    logger.log(0)

    for act_ind, act in enumerate(actions):
        hold_objects = names[act_ind + 1:]
        drop_object = names[act_ind]
        if act_ind == len(actions) -1 and hold_last_action:
            logger.hold_drop_execute(hold_objects, drop_object, 1)
        else:
            logger.hold_drop_execute(hold_objects, drop_object, sim_steps)
        logger.log(act_ind + 1, hold_objects)

    data, images, masks = logger.get_logs()
    images = images / 255.

    return data, images, masks

def swap_image(im):
    return np.swapaxes(np.swapaxes(im, 0, 2), 0, 1)

class MPC:
    polygons = ['cube'] #, 'horizontal_rectangle', 'tetrahedron']
    polygon_map = {ind: ply for ind, ply in enumerate(polygons)}

    def __init__(self, model):
        self.model = model


    def run(self, goal_image, mpc_h=6):

        goal_image = goal_image.astype(np.float32) / 255
        #import pdb; pdb.set_trace()
        self.model.gen_image(goal_image,
                            '/home/jcoreyes/objects/rlkit/examples/iodine/test_mpc/goal.png')


        actions_lst = []
        for i in range(mpc_h):
            best_action = self.single_step_mpc(goal_image, i, actions_lst)
            actions_lst.append(best_action)
            print(i)


    def single_step_mpc(self, goal_image, mpc_itr, actions_so_far):
        # goal image is (imsize, imsize, ch)



        actions = self.sample_actions(100)

        asset_path = '/home/jcoreyes/objects/object-oriented-prediction/o2p2/data/stl'

        true_actions = np.array([ [0, -.75,0,0, 0,0,1,0, .4, .75,.75,0],
			[0, -.75,0,1, 0,0,1,0, .4, .25,.75,.25],
			[0,  .75,0,0, 0,0,1,0, .4, .5,.25,1],
			[0,  .75,0,1, 0,0,1,0, .4, 1,.25,.5]])
        #actions.append(true_actions[mpc_itr])
        actions = [true_actions[mpc_itr, :]]
        all_images = []
        for action in actions:
            a = [action[0], action[1:4], action[4:8], action[8], action[9:12]]
            #if mpc_itr == 1:
            #    a = [0, [-.75,0,1], [0,0,1,0], .4, np.array([.25,.75,.25])]
            data, images, masks = execute_actions(actions_so_far + [a], self.polygons, asset_path, sim_steps=2000,
                                                  hold_last_action=True)
            # images is (2, imsize, imsize, 3)
            all_images.append(images)

        initial_states = np.swapaxes(np.stack([x[-1] for x in all_images]), 1, 3)
        initial_states = np.swapaxes(initial_states, 2, 3)
        #if mpc_itr == 1:
        #    import pdb; pdb.set_trace()
        input = ptu.from_numpy(np.expand_dims(initial_states, 1).repeat(5, axis=1))

        final_recon = self.model.predict(input, seedsteps=11, bs=1)

        pred = ptu.get_numpy(torch.clamp(final_recon, 0, 1)) # (bs, ch, imsize, imsize)
        pred = np.swapaxes(np.swapaxes(pred, 1, 3), 1, 2) # (bs, imsize, imsize, ch)

        mse = np.power(np.expand_dims(goal_image, 0), pred).mean((1, 2, 3))

        best_idx = mse.argmin()


        best_action_initial_img = all_images[best_idx][-1]
        best_recon = pred[best_idx]
        action = actions[best_idx]
        a = [action[0], action[1:4], action[4:8], action[8], action[9:12]]
        _, best_executed_action_img, _ = execute_actions(actions_so_far + [a], self.polygons, asset_path, sim_steps=2000)
        #import pdb; pdb.set_trace()
        comparison = np.stack([goal_image, best_action_initial_img, best_executed_action_img[-1], best_recon, ])

        save_dir = '/home/jcoreyes/objects/rlkit/examples/iodine/test_mpc/%d.png' % mpc_itr
        #import pdb; pdb.set_trace()
        save_image(ptu.from_numpy(np.swapaxes(np.swapaxes(comparison, 1, 3), 2, 3)).cpu(), save_dir, nrow=4)

        return a

    def sample_actions(self, n_actions):
        #
        #
        # ## ply_ind, pos, axangle, scale, rgb
        # actions = [[0, [-.75, 0, 0], [0, 0, 1, 0], .4, [.75, .75, 0]],
        #            [0, [-.75, 0, 1], [0, 0, 1, 0], .4, [.25, .75, .25]],
        #            [0, [.75, 0, 0], [0, 0, 1, 0], .4, [.5, .25, 1]],
        #            [0, [.75, 0, 1], [0, 0, 1, 0], .4, [1, .25, .5]],
        #            [1, [0, 0, 2], [0, 0, 1, 0], .4, [.85, .25, 0]],
        #            [2, [0, 0, 3], [0, 0, 1, math.pi / 4], .4, [0, .75, .75]],

        ply_idx = np.random.randint(0, len(self.polygons), size=(n_actions, 1))
        pos = np.random.uniform(low=[-1, 0, 0], high=[1, 0, 3], size=(n_actions, 3))
        axangle = np.random.uniform(low=[0, 0, 1, 0], high=[0, 0, 1, 0], size=(n_actions, 4))
        scale = np.random.uniform(low=0.4, high=0.4, size=(n_actions, 1))
        #rgb = np.random.uniform(low=[0, .25, 0], high=[1, .75, 1], size=(n_actions, 3))
        colors = [ [.75,.75,0],
                   [.25, .75, .25],
                   [.5, .25, 1],
                   [1, .25, .5],
                   [.85, .25, 0],
                   [0, .75, .75]
                   ]
        rgb = np.random.randint(0, len(colors), size=(n_actions))
        rgb = np.array([colors[x] for x in rgb])
        actions = np.concatenate([ply_idx, pos, axangle, scale, rgb], -1)
        actions = [x for x in actions]

        return actions




def main(variant):


    train_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock.h5'
    test_path = '/home/jcoreyes/objects/RailResearch/BlocksGeneration/rendered/fiveBlock.h5'

    # train_data = load_dataset(train_path, train=True)
    # test_data = load_dataset(test_path, train=False)
    # n_frames = 35
    # imsize = train_data.shape[-1]
    # T = variant['vae_kwargs']['T']
    # K = variant['vae_kwargs']['K']
    # rep_size = variant['vae_kwargs']['representation_size']
    # # t_sample = np.array([0, 0, 0, 0, 0, 10, 15, 20, 25, 30])
    # t_sample = np.array([0, 0, 0, 34, 34])
    # train_data = train_data.reshape((n_frames, -1, 3, imsize, imsize)).swapaxes(0, 1)[:1000, t_sample]
    # test_data = test_data.reshape((n_frames, -1, 3, imsize, imsize)).swapaxes(0, 1)[:50, t_sample]
    #
    # train_goals = train_data[:, -1]
    # test_goals = test_data[:, -1]

    model_file = '/home/jcoreyes/objects/rlkit/output/04-15-iodine-blocks-physics-noseed/04-15-iodine-blocks-physics_noseed_2019_04_15_12_19_02_0000--s-94700/params.pkl'
    #model_file = '/home/jcoreyes/objects/rlkit/output/04-09-iodine-blocks-physics/04-09-iodine-blocks-physics_2019_04_09_21_12_02_0000--s-9028/params.pkl'
    model = pickle.load(open(model_file, 'rb'))
    model.cuda()
    mpc = MPC(model)
    import imageio
    goal_image = imageio.imread('/home/jcoreyes/objects/object-oriented-prediction/o2p2/planning/executed/mjc_2.png')
    mpc.run(goal_image)
    #  m.to(ptu.device)


    # save_period = variant['save_period']
    # for epoch in range(variant['num_epochs']):
    #     should_save_imgs = (epoch % save_period == 0)
    #     t.train_epoch(epoch, batches=train_data.shape[0]//variant['algo_kwargs']['batch_size'])
    #     t.test_epoch(epoch, save_vae=True, train=False, record_stats=True, batches=1,
    #                  save_reconstruction=should_save_imgs)
    #     t.test_epoch(epoch, save_vae=False, train=True, record_stats=False, batches=1,
    #                  save_reconstruction=should_save_imgs)
    #     #if should_save_imgs:
    #     #    t.dump_samples(epoch)
    # logger.save_extra_data(m, 'vae.pkl', mode='pickle')


if __name__ == "__main__":
    variant = dict(
        vae_kwargs = dict(
            imsize=64,
            representation_size=128,
            input_channels=3,
            decoder_distribution='gaussian_identity_variance',
            beta=1,
            K=5,
            T=5,
        ),
        algo_kwargs = dict(
            gamma=0.5,
            batch_size=8,
            lr=1e-4,
            log_interval=0,
        ),
        num_epochs=10000,
        algorithm='VAE',
        save_period=5,
        physics=True
    )


    run_experiment(
        main,
        exp_prefix='iodine-blocks-mpc',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



