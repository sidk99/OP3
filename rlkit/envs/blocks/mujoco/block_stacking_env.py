import os
import argparse
import pickle
import random
import copy
import pdb

import colorsys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import mujoco_py as mjc

from rlkit.envs.blocks.mujoco.logger import Logger
import rlkit.envs.blocks.mujoco.utils as utils
from rlkit.envs.blocks.mujoco.XML import XML

#RV
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
        image = images[0] #/255
        # print(image.shape)
        return image

    def sample_action(self):
        ply = random.choice(self.polygons)

        pos = utils.uniform(*self.drop_bounds['pos'])
        # pos[-1] = obj_num

        if 'horizontal' in ply:
            axis = [1, 0, 0]
        else:
            axis = [0, 0, 1]
        axangle = utils.random_axangle(axis=axis)
        axangle[-1] = 0

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

    def sample_action_gaussian(self, mean, std):
        #ply = random.choice(self.polygons)

        ply_t = .1
        ply_p = (mean[:3] + ply_t) / (mean[:3] + ply_t).sum()
        ply = np.random.choice(self.polygons, p=ply_p)


        std = np.maximum(std, 0.01)
        random_a = np.random.normal(mean, std)



        pos = np.clip(random_a[3:6], [x[0] for x in self.drop_bounds['pos']],
                      [x[1] for x in self.drop_bounds['pos']])

        # pos[-1] = np.clip(pos[-1], -.4, 0)
        # pos[-1] = 2

        if 'horizontal' in ply:
            axis = [1, 0, 0]
        else:
            axis = [0, 0, 1]
        axangle = utils.random_axangle(axis=axis)

        #axangle[-1] = random_a[9]
        axangle[-1] = 0

        if 'horizontal' in ply:
            axangle[-1] = 0

        scale = utils.uniform(*self.settle_bounds['scale'])
        #rgba = self.sample_rgba_from_hsv(*self.settle_bounds['hsv'])

        # rgba = np.clip(random_a[-3:], [x[0] for x in self.settle_bounds['hsv']],
        #               [x[1] for x in self.settle_bounds['hsv']])
        rgba = np.clip(random_a[-3:], 0, 1)
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

    ##Tries an action and returns the direct observation of the action (e.g. the block in the air)
    ##  but does not actually take a step in the environment
    def try_action(self, an_action):
        xml = XML(self.asset_path)
        # Note: We need to recreate the entire scene
        for ind, prev_action in enumerate(self.xml_actions_taken):  # Adding previous actions
            prev_action['pos'][-1] = ind * 2
            xml.add_mesh(**prev_action)

        xml_action = self.model_action_to_xml_action(an_action)
        new_name = xml.add_mesh(**xml_action)  # Note name is name of action (name of block dropped)
        new_names = self.names + [new_name]

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)

        logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)
        for act_ind, act in enumerate(new_names[:-1]):
            logger.hold_drop_execute(new_names[act_ind+1:], new_names[act_ind], self.settle_steps)
        logger.log(0)

        original_logger = self.logger
        self.logger = logger
        obs = self.get_observation()
        self.logger = original_logger
        return obs

    def try_actions(self, actions):
        xml = XML(self.asset_path)
        # Note: We need to recreate the entire scene
        for ind, prev_action in enumerate(self.xml_actions_taken):  # Adding previous actions
            prev_action['pos'][-1] = ind * 2
            xml.add_mesh(**prev_action)

        #xml_action = self.model_action_to_xml_action(an_action)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)

        logger = Logger(xml, sim, steps=self.max_num_objects_dropped + 1, img_dim=self.img_dim)
        for act_ind, act in enumerate(self.names):
            logger.hold_drop_execute(self.names[act_ind+1:], self.names[act_ind], self.settle_steps)

        sim_data = sim.get_state()

        #import pdb; pdb.set_trace()

        xml = XML(self.asset_path)
        # Note: We need to recreate the entire scene
        for ind, prev_action in enumerate(self.xml_actions_taken):  # Adding previous actions
            prev_action['pos'][-1] = ind * 2
            xml.add_mesh(**prev_action)


        # Add all new blocks but outside of scene at different heights so they don't collide
        old_pos = []
        for i, action in enumerate(actions):
            xml_action = self.model_action_to_xml_action(action)
            old_pos.append(copy.deepcopy(xml_action['pos']))
            xml_action['pos'][-1] = (6 + i)

            new_name = xml.add_mesh(**xml_action)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        sim = mjc.MjSim(model)
        obs_lst = []
        old_sim = self.logger.sim
        self.logger.sim = sim

        for i, action in enumerate(actions):
            # set old data
            if len(self.xml_actions_taken) > 0:
                sim.data.qpos[:sim_data.qpos.shape[0]] = sim_data.qpos
                sim.data.qvel[:sim_data.qvel.shape[0]] = sim_data.qvel
                #sim.forward()
                # set new data
            # set new block data
            block_idx = (len(self.xml_actions_taken) + i)  * 7

            air_pos = copy.deepcopy(sim.data.qpos[block_idx:block_idx+3])
            #import pdb; pdb.set_trace()
            sim.data.qpos[block_idx:block_idx+3] = old_pos[i]

            sim.step()
            obs = self.logger.log_image(0)
            obs_lst.append(obs)

            # set it back
            sim.data.qpos[block_idx:block_idx + 3] = air_pos
        #self.logger = original_logger
        self.logger.sim = old_sim

        return obs_lst


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

    def compute_accuracy(self, true_data):
        state = self.logger.get_state()
        return self.compare_matching(state, true_data['data']), state

    def compare_matching(self, data, mjc_data, threshold=0.2):
        # data is env, mjc_data is target
        # data = data.val[0].val
        mjc_data = copy.deepcopy(mjc_data)

        max_err = -float('inf')
        # pdb.set_trace()
        for pred_name, pred_datum in data.items():
            err, mjc_match, err_pos, err_rgb = self._best_obj_match(pred_datum, mjc_data)
            del mjc_data[mjc_match]

            # print(err)
            if err > max_err:
                max_err = err
                max_pos = err_pos
                max_rgb = err_rgb

            if len(mjc_data) == 0:
                break

        correct = max_err < threshold
        return correct, max_pos, max_rgb

    def _best_obj_match(self, pred, targs):
        def np_mse(x1, x2):
            return np.square(x1 - x2).mean()

        pos = pred['qpos'][:3]
        rgb = pred['rgba']

        best_err = float('inf')
        for obj_name, obj_data in targs.items():
            obj_pos = obj_data['xpos'][-1]
            obj_rgb = obj_data['xrgba'][-1]

            pos_err = np_mse(pos, obj_pos)
            rgb_err = np_mse(rgb, obj_rgb)
            err = pos_err + rgb_err

            if err < best_err:
                best_err = err
                best_obj = obj_name
                best_pos = pos_err
                best_rgb = rgb_err

        return best_err, best_obj, best_pos, best_rgb



def sanity_check_accuracy():
    env1 = BlockEnv(5)
    actions = []
    for i in range(3):
        action = env1.sample_action()
        env1.step(action)
        actions.append(action)

    env2 = BlockEnv(5)
    for i in range(3):
        action = env2.sample_action_gaussian(np.array(actions[i]), 0.01)
        env2.step(action)

    true_data = env1.logger.get_state()
    for obj_name, obj_data in true_data.items():
        true_data[obj_name]['xpos'] = np.array([true_data[obj_name]['qpos'][:3]])
        true_data[obj_name]['xrgba'] = np.array([true_data[obj_name]['rgba'][:3]])
    true_data = {'data': true_data}
    tmp = env2.compute_accuracy(true_data)
    print(tmp)

    cur_fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4 * 6, 1 * 6))
    axes[0].imshow(env1.get_observation()/255, interpolation='nearest')
    axes[1].imshow(env2.get_observation()/255, interpolation='nearest')
    cur_fig.savefig("HELLO")


def check_bugs_in_try_action():
    num_blocks = 5
    env = BlockEnv(num_blocks)
    cur_fig, axes = plt.subplots(nrows=2, ncols=num_blocks, figsize=(num_blocks * 6, 2 * 6))
    for i in range(num_blocks):
        action = env.sample_action()
        axes[0, i].imshow(env.try_action(action)/255, interpolation='nearest')
        axes[1, i].imshow(env.step(action) / 255, interpolation='nearest')
    cur_fig.savefig("HELLO")

def check_bugs_in_try_actionS():
    num_actions = 5
    env = BlockEnv(4)
    cur_fig, axes = plt.subplots(nrows=2, ncols=num_actions, figsize=(num_actions * 6, 2 * 6))
    actions = [env.sample_action() for _ in range(num_actions)]
    results = env.try_actions(actions)

    for i in range(num_actions):
        axes[0, i].imshow(env.try_action(actions[i]) / 255, interpolation='nearest')
        axes[1, i].imshow(results[i] / 255, interpolation='nearest')

    cur_fig.savefig("HELLO")



if __name__ == '__main__':
    # sanity_check_accuracy()
    # check_bugs_in_try_action()
    check_bugs_in_try_actionS()

    # cur_fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(4 * 6, 1 * 6))
    #
    # myenv = BlockEnv(5)
    # axes[0].imshow(myenv.get_observation(), interpolation='nearest')
    # # cur_fig.savefig("HELLO")
    # # plt.show()
    #
    # # the_input = input("Suffix: ")
    # tmp = myenv.step(myenv.sample_action())
    # axes[1].imshow(tmp)
    # obs = myenv.try_actions([myenv.sample_action() for i in range(4)])
    #
    # for i in range(4):
    #     axes[i+2].imshow(obs[i])
    #
    #
    # cur_fig.savefig("HELLO")

