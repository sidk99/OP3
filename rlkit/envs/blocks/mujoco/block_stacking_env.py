import os
import argparse
import pickle
import random

import colorsys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import mujoco_py as mjc

from rlkit.envs.blocks.mujoco.logger import Logger
import rlkit.envs.blocks.mujoco.utils as utils
from rlkit.envs.blocks.mujoco.XML import XML


# polygons = ['cube', 'horizontal_rectangle', 'tetrahedron']

# num_objects = range(args.min_objects, args.max_objects + 1)

## bounds for objects that start on the ground plane
# settle_bounds = {
#             'pos':   [ [-.5, .5], [-.5, 0], [1, 2] ],
#             'hsv': [ [0, 1], [0.5, 1], [0.5, 1] ],
#             'scale': [ [0.4, 0.4] ],
#             'force': [ [0, 0], [0, 0], [0, 0] ]
#           }
#
# ## bounds for the object to be dropped
# drop_bounds = {
#             'pos':   [ [-1.75, 1.75], [-.5, 0], [0, 3] ],
#           }

## folder with object meshes
# asset_path = os.path.join(os.getcwd(), '../data/stl/')

# utils.mkdir(args.output_path)

# metadata = {'polygons': polygons, 'max_steps': args.drop_steps_max,
#             'min_objects': min(num_objects),
#             'max_objects': max(num_objects)}
# pickle.dump( metadata, open(os.path.join(args.output_path, 'metadata.p'), 'wb') )


class BlockEnv():
    def __init__(self, max_num_objects_dropped):
        self.asset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stl/')
        self.img_dim = 64
        self.polygons = ['cube', 'horizontal_rectangle', 'tetrahedron']
        self.settle_bounds = {
            'pos':   [ [-.5, .5], [-.5, 0], [1, 2] ],
            'hsv': [ [0, 1], [0.5, 1], [0.5, 1] ],
            'scale': [ [0.4, 0.4] ],
            'force': [ [0, 0], [0, 0], [0, 0] ]
          }

        self.drop_bounds = {
            'pos':   [ [-1.75, 1.75], [-.5, 0], [0, 3] ],
          }

        self.xml = XML(self.asset_path)

        xml_str = self.xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)

        self.max_num_objects_dropped = max_num_objects_dropped
        self.logger = Logger(self.xml, sim, steps=max_num_objects_dropped + 1, img_dim=self.img_dim)
        self.logger.log(0)

        self.xml_actions_taken = []
        self.names = []
        self.env_step = 0
        self.settle_steps = 2000

    def reset(self):
        xml = XML(self.asset_path)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)
        self.logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)
        self.logger.log(0)

        self.xml_actions_taken = []
        self.names = []
        self.env_step = 0

        return self.get_observation()


    def get_observation(self):
        data, images, masks = self.logger.get_logs()
        # print(images.shape, masks.keys())
        image = images[0]/255
        # print(image.shape)
        return image

    def sample_action(self):
        ply = random.choice(self.polygons)

        pos = utils.uniform(*self.settle_bounds['pos'])
        # pos[-1] = obj_num

        if 'horizontal' in ply:
            axis = [1, 0, 0]
        else:
            axis = [0, 0, 1]
        axangle = utils.random_axangle(axis=axis)
        scale = utils.uniform(*self.settle_bounds['scale'])
        rgba = self.sample_rgba_from_hsv(*self.settle_bounds['hsv'])
        xml_action = {
            'polygon': ply,
            'pos': pos,
            'axangle': axangle,
            'scale': scale,
            'rgba': rgba
        }
        # print(xml_action)

        return self.xml_action_to_model_action(xml_action)

    def get_obs_size(self):
        return (self.img_dim, self.img_dim)

    def get_actions_size(self):
        return (15)

    def step(self, an_action):
        #an_action should contain one_hot of polygon[3], pos[3], axangle[4], scale[1], rgba[3]
        #Total size: 3+3+4+1+4 = 15

        xml = XML(self.asset_path)
        #Note: We need to recreate the entire scene
        for ind, prev_action in enumerate(self.xml_actions_taken): # Adding previous actions
            prev_action['pos'][-1] = ind*2
            xml.add_mesh(**prev_action)

        xml_action = self.model_action_to_xml_action(an_action)
        # print("Action to take: ", xml_action)
        new_name = xml.add_mesh(**xml_action) #Note name is name of action (name of block dropped)

        self.names.append(new_name)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)

        logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)
        # logger.log(0)

        for act_ind, act in enumerate(self.names):
            logger.hold_drop_execute(self.names[act_ind+1:], self.names[act_ind], self.settle_steps)
            # logger.log(act_ind+1)
        # logger.hold_drop_execute(self.names, new_name, 1)
        # logger.log(len(self.xml_actions_taken)+1)

        # print(self.xml_actions_taken)
        self.logger = logger
        self.logger.log(0)

        ##Update state information
        self.xml_actions_taken.append(xml_action)
        # self.names.append(new_name)

        return self.get_observation()


    #############Internal functions#########
    def model_action_to_xml_action(self, model_action):
        # an_action should contain one_hot of polygon[3], pos[3], axangle[4], scale[1], rgba[4]
        # Total size: 3+3+4+1+4 = 15
        #an_action should be of size [15]
        # print(np.where(model_action == 1))
        # print(np.where(model_action == 1)[0], "HI", np.where(model_action == 1)[0][0])
        ans = {
            "polygon": self.polygons[np.where(model_action == 1)[0][0]],
            "pos": model_action[3:6],
            "axangle": model_action[6:10],
            "scale": 0.4,
            "rgba": np.concatenate([model_action[10:], np.array([1])])
        }
        # ans = {
        #     "polygon": self.polygons[int(model_action[0])],
        #     "pos": model_action[1:4],
        #     "axangle": model_action[4:8],
        #     "scale": model_action[8],
        #     "rgba": model_action[9:]
        # }
        return ans

    def xml_action_to_model_action(self, xml_action):
        num_type_polygons = len(self.polygons)
        total_size_of_array = 13 #num_type_polygons+3+4+1+4  #polygon[3], pos[3], axangle[4], scale[1], rgba[4]
        # print(total_size_of_array)
        ans = np.zeros(total_size_of_array)

        poly_name = xml_action['polygon']
        val = self.polygons.index(poly_name)
        ans[val] = 1

        for i in range(len(xml_action["pos"])):
            ans[num_type_polygons + i] = xml_action["pos"][i]

        for i in range(len(xml_action["axangle"])):
            ans[num_type_polygons + 3 + i] = xml_action["axangle"][i]

        #ans[num_type_polygons + 3 + 4] = xml_action["scale"]

        for i in range(3):
            ans[num_type_polygons +3+4+ i] = xml_action["rgba"][i]

        # TODO make into sids 13 for now since model was trained on size 13 version
        return ans

    def sample_rgba_from_hsv(self, *hsv_bounds):
        hsv = utils.uniform(*hsv_bounds)
        rgba = list(colorsys.hsv_to_rgb(*hsv)) + [1]
        return rgba


if __name__ == '__main__':
    cur_fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(4 * 6, 1 * 6))

    myenv = BlockEnv(5)
    obs = []
    obs.append(myenv.get_observation())
    axes[0].imshow(obs[-1], interpolation='nearest')
    # cur_fig.savefig("HELLO")
    # plt.show()

    # the_input = input("Suffix: ")

    for i in range(4):
        an_action = myenv.sample_action()
        myenv.step(an_action)
        tmp = myenv.get_observation()
        # print(np.max(tmp), np.min(tmp), np.mean(tmp))
        axes[i+1].imshow(tmp, interpolation='nearest')
        # for k in range(5):
        #     axes[i+1, k].imshow(tmp[k], interpolation='nearest')

    cur_fig.savefig("HELLO")

