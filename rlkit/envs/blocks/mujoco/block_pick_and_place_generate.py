import os
import pdb
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import rlkit.envs.blocks.mujoco.utils.data_generation_utils as dgu

from rlkit.envs.blocks.mujoco.block_pick_and_place import BlockPickAndPlaceEnv

def createSingleSim(args):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    num_blocks = np.random.randint(args.min_num_objects, args.max_num_objects+1)
    myenv = BlockPickAndPlaceEnv(num_blocks, args.num_colors, args.img_dim, include_z=False)
    myenv.img_dim = args.img_dim
    imgs = []
    acs = []
    imgs.append(myenv.get_observation())
    rand_float = np.random.uniform()
    for t in range(args.num_frames-1):
        if rand_float < args.force_pick:
            ac = myenv.sample_action('pick_block')
        elif rand_float < args.force_pick + args.force_place:
            ac = myenv.sample_action('place_block')
        else:
            ac = myenv.sample_action()
        imgs.append(myenv.step(ac))
        acs.append(ac)
    acs.append(myenv.sample_action(None))
    return np.array(imgs), np.array(acs)


"""
python rlkit/envs/blocks/mujoco/block_pick_and_place.py -f data/pickplace50k.h5 -nmin 3 -nax 4 -nf 2 -ns 50000 -fpick 0.3 -fplace 0.4
"""

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None)
    parser.add_argument('-nmin', '--min_num_objects', type=int, default=3)
    parser.add_argument('-nax', '--max_num_objects', type=int, default=3)
    parser.add_argument('-i', '--img_dim', type=int, default=64)
    parser.add_argument('-nf', '--num_frames', type=int, default=51)
    parser.add_argument('-ns', '--num_sims', type=int, default=5)
    parser.add_argument('-mi', '--make_images', type=bool, default=False)
    parser.add_argument('-c', '--num_colors', type=int, default=None)
    parser.add_argument('-fpick', '--force_pick', type=float, default=0.3)
    parser.add_argument('-fplace', '--force_place', type=float, default=0.2)
    parser.add_argument('--output_path', default='', type=str,
                        help='path to save images')
    parser.add_argument('-p', '--num_workers', type=int, default=1)

    args = parser.parse_args()
    print(args)

    info = {}
    info["min_num_objects"] = args.min_num_objects
    info["max_num_objects"] = args.max_num_objects
    info["img_dim"] = args.img_dim
    info["num_frames"] = args
    single_sim_func = createSingleSim
    env = BlockPickAndPlaceEnv(1, 1, args.img_dim)
    ac_size = env.get_actions_size()
    obs_size = env.get_obs_size()
    dgu.createMultipleSims(args, obs_size, ac_size, single_sim_func, num_workers=int(args.num_workers))

    dgu.hdf5_to_image(args.filename)
    for i in range(10):
        tmp = os.path.join(args.output_path, "data/imgs/training/{}/features".format(str(i)))
        dgu.make_gif(tmp, "animation.gif")
        # tmp = os.path.join(args.output_path, "imgs/training/{}/groups".format(str(i)))
        # dgu.make_gif(tmp, "animation.gif")


    ##Running and rendering example
    # b = BlockPickAndPlaceEnv(4, None, 64)
    # for i in range(10000):
    #     # pdb.set_trace()
    #     # b.step()
    #     # if i == 2000:
    #     #     cur_pos = b.get_block_info("cube_0")["pos"]
    #     #     ac = list(cur_pos) + [-0.5, -0.5, 1]
    #     #     b.step(ac)
    #     if i < 2000:# and i %100 == 0:
    #         # cur_pos = b.get_block_info("cube_0")["pos"]
    #         # ac = list(cur_pos) + [-0.5, -0.5, 1]
    #         b.step(b.sample_action("pick_block"))
    #     b.viewer.render()

    #Plotting images
    # cur_fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(4 * 6, 1 * 6))
    #
    # myenv = BlockPickAndPlaceEnv(4, None, 64)
    # obs = []
    # obs.append(myenv.get_observation())
    # axes[0].imshow(obs[-1], interpolation='nearest')
    # # cur_fig.savefig("HELLO")
    # # plt.show()
    #
    # # the_input = input("Suffix: ")
    #
    # for i in range(4):
    #     # an_action = myenv.sample_action()
    #     # tmp_name = myenv.step(an_action)
    #     # while (not tmp_name):
    #     #     an_action = myenv.sample_action()
    #     #     tmp_name = myenv.step(an_action)
    #     ac = myenv.sample_action(True)
    #     tmp = myenv.step(ac)
    #     print(tmp.shape, np.max(tmp), np.min(tmp), np.mean(tmp))
    #     # print(np.max(tmp), np.min(tmp), np.mean(tmp))
    #     axes[i+1].set_title(ac)
    #     axes[i + 1].imshow(tmp)#, interpolation='nearest')
    #     # for k in range(5):
    #     #     axes[i+1, k].imshow(tmp[k], interpolation='nearest')
    # cur_fig.savefig("HELLO")