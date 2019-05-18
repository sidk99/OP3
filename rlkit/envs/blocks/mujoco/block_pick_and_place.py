import os
import pdb
import numpy as np
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import rlkit.envs.blocks.mujoco.utils.data_generation_utils as dgu
import mujoco_py
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import rlkit.envs.blocks.mujoco.contacts as contacts

MODEL_XML_BASE = """
<mujoco>
    <asset>
        <material name='wall_visible' rgba='.9 .9 .9 1' specular="0" shininess="0"  emission="0.25"/>
        <material name='wall_invisible' rgba='.9 .9 .9 0' specular="0" shininess="0" emission="0.25"/>
       {}
       {}
    </asset>
    <worldbody>
        <camera name='fixed' pos='0 -8 6' euler='-300 0 0' fovy='55'/>
        <light diffuse='1.5 1.5 1.5' pos='0 -7 8' dir='0 -1 -1'/>  
        <light diffuse='1.5 1.5 1.5' pos='0 -7 6' dir='0 -1 -1'/>  
        <geom name='wall_floor' type='plane' pos='0 0 0' euler='0 0 0' size='20 10 0.1' material='wall_visible' 
        condim='3' />
        {}
    </worldbody>
</mujoco>
"""
#60 = -300
# <geom name='wall_front'  type='box' pos='0 -5 0' euler='0 0 0' size='10 0.1 4' material='wall_visible'/>
# <geom name='wall_left'  type='box' pos='-5 0 0' euler='0 0 0' size='0.1 10 4' material='wall_visible'/>
# <geom name='wall_right'  type='box' pos='5 0 0' euler='0 0 0' size='0.1 10 4' material='wall_visible'/>
# <geom name='wall_back'  type='box' pos='0 5 0' euler='0 0 0' size='10 0.1 4' material='wall_visible'/>

# <body name="floor" pos="0 0 0.025">
#     <geom size="3.0 3.0 0.02" rgba="0 1 0 1" type="box"/>
#     <camera name='fixed' pos='0 -8 8' euler='45 0 0'/>
# </body>

#Red, Lime, Blue, Yellow, Cyan, Magenta, Black, White
COLOR_LIST = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [1, 1, 1], [255, 255, 255]]
def pickRandomColor(an_int):
    if an_int is None:
        return np.random.uniform(low=0.0, high=1.0, size=3)
    tmp = np.random.randint(0, an_int)
    return np.array(COLOR_LIST[tmp])/255
import copy

class BlockPickAndPlaceEnv():
    def __init__(self, num_objects, num_colors, img_dim, include_z, random_initialize=False, view=False):
        # self.asset_path = os.path.join(os.getcwd(), '../data/stl/')
        self.asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/stl/')
        # self.asset_path = os.path.join(os.path.realpath(__file__), 'data/stl/')
        # self.asset_path = '../data/stl/'
        self.img_dim = img_dim
        self.polygons = ['cube', 'horizontal_rectangle', 'tetrahedron'][:2]
        self.num_colors = num_colors
        self.num_objects = num_objects
        self.view = view
        self.internal_steps_per_step = 1000
        self.drop_heights = 5
        self.bounds = {'x_min':-2.5, 'x_max':2.5, 'y_min':-2, 'y_max':1, 'z_min':0.05, 'z_max': 2.2}
        self.include_z = include_z

        self.names = []
        self.blocks = []

        if random_initialize:
            self.reset()

    ####Env initialization functions
    def get_unique_name(self, polygon):
        i = 0
        while '{}_{}'.format(polygon, i) in self.names:
            i += 1
        name = '{}_{}'.format(polygon, i)
        self.names.append(name)
        return name

    def add_mesh(self, polygon, pos, quat, rgba):
        name = self.get_unique_name(polygon)
        self.blocks.append({'name': name, 'polygon': polygon, 'pos': np.array(pos), 'quat': np.array(quat), 'rgba': rgba,
                            'material': name})

    def get_asset_material_str(self):
        asset_base = '<material name="{}" rgba="{}" specular="0" shininess="0" emission="0.25"/>'
        asset_list = [asset_base.format(a['name'], self.convert_to_str(a['rgba'])) for a in self.blocks]
        asset_str = '\n'.join(asset_list)
        return asset_str

    def get_asset_mesh_str(self):
        asset_base = '<mesh name="{}" scale="0.6 0.6 0.6" file="{}"/>'
        asset_list = [asset_base.format(a['name'], os.path.join(self.asset_path, a['polygon'] + '.stl'))
                      for a in self.blocks]
        asset_str = '\n'.join(asset_list)
        return asset_str

    def get_body_str(self):
        body_base = '''
          <body name='{}' pos='{}' quat='{}'>
            <joint type='free' name='{}'/>
            <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}' condim='3'/>
          </body>
        '''
        # body_base = '''
        #   <body name='{}' pos='{}' quat='{}'
        #     <joint type='free' name='{}' damping='0'/>
        #     <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}' condim='1' friction='1 1 1'/>
        #   </body>
        # '''
        # '''<geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}' condim="6" friction="1 1 1"/>'''
        body_list = [body_base.format(m['name'], self.convert_to_str(m['pos']),
                                      self.convert_to_str(m['quat']), m['name'],
                                      m['name'], m['name'], m['material']) for i, m in enumerate(self.blocks)]
        body_str = '\n'.join(body_list)
        return body_str

    def convert_to_str(self, an_iterable):
        tmp = ""
        for an_item in an_iterable:
            tmp += str(an_item) + " "
        # tmp = " ".join(str(an_iterable))
        return tmp[:-1]

    def get_random_pos(self, height=None):
        x = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
        y = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
        if height is None:
            z = np.random.uniform(1, self.bounds['z_max'])
        else:
            z = height
        return np.array([x, y, z])

    def get_random_rbga(self, num_colors):
        rgb = list(pickRandomColor(num_colors))
        return rgb + [1]

    def initialize(self, use_cur_pos):
        tmp = MODEL_XML_BASE.format(self.get_asset_mesh_str(), self.get_asset_material_str(), self.get_body_str())
        model = load_model_from_xml(tmp)
        self.sim = MjSim(model)
        if self.view:
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

        self.sim_state = self.sim.get_state()
        self.get_starting_step(use_cur_pos)

    def get_starting_step(self, use_cur_pos):
        prev_positions = {}
        for i, aname in enumerate(self.names):
            if use_cur_pos:
                prev_positions[aname] = self.get_block_info(aname)["pos"]
            self.add_block(aname, [-5+i, -5+i, -5])

        for aname in self.names:
            if use_cur_pos:
                tmp_pos = prev_positions[aname]
                # print(aname, tmp_pos)
            else:
                tmp_pos = self.get_random_pos(self.drop_heights)
            self.add_block(aname, tmp_pos)
            for i in range(self.internal_steps_per_step):
                self.internal_step()
                if self.view:
                    self.viewer.render()
            self.sim_state = self.sim.get_state()

    def move_blocks_side(self):
        # Move blocks to either side
        z = 1
        side_pos = [
            [-2.4, -2, z],
            [-2.4, 0, z],
            [2.4, -2, z],
            [2.4, 0, z],
                            ]

        true_actions = []
        for i, block in enumerate(self.names):
            pos = self.get_block_info(block)
            self.add_block(block, side_pos[i])
            true_actions.append([side_pos[i] + pos])

        for i in range(100):
            self.sim.step()

        return true_actions

    ####Env internal step functions
    def add_block(self, ablock, pos):
        #pos (x,y,z)
        self.set_block_info(ablock, {"pos": pos})
        # self.sim.set_state(self.sim_state)

    def pick_block(self, pos):
        block_name = None
        for a_block in self.names:
            if self.intersect(a_block, pos):
                block_name = a_block

        if block_name is None:
            return False

        #PICK_LOC = np.array([0, 0, 5])
        #info = {"pos":PICK_LOC}
        #self.set_block_info(block_name, info)
        # self.sim.set_state(self.sim_state)
        return block_name

    def intersect(self, a_block, pos):
        #Threshold
        THRESHHOLD = 0.2
        cur_pos = self.get_block_info(a_block)["pos"]
        return np.max(np.abs(cur_pos - pos)) < THRESHHOLD

    def get_block_info(self, a_block):
        info = {}
        info["poly"] = a_block[:-2]
        info["pos"] = np.copy(self.sim.data.get_body_xpos(a_block)) #np array
        info["quat"] = np.copy(self.sim.data.get_body_xquat(a_block))
        info["vel"] = np.copy(self.sim.data.get_body_xvelp(a_block))
        info["rot_vel"] = np.copy(self.sim.data.get_body_xvelr(a_block))
        return info

    def set_block_info(self, a_block, info):
        #import pdb; pdb.set_trace()
        # print(a_block, info)
        # print("Setting state: {}, {}".format(a_block, info))
        start_ind = self.sim.model.get_joint_qpos_addr(a_block)[0]
        if "pos" in info:
            self.sim_state.qpos[start_ind:start_ind+3] = np.array(info["pos"])
        #if "quat" in info:
        #    self.sim_state.qpos[start_ind+3:start_ind+7] = info["quat"]
        #else:
        self.sim_state.qpos[start_ind + 3:start_ind + 7] = np.array([1, 0, 0, 0])

        start_ind = self.sim.model.get_joint_qvel_addr(a_block)[0]
        if "vel" in info:
            self.sim_state.qvel[start_ind:start_ind + 3] = info["vel"]
        else:
            self.sim_state.qvel[start_ind:start_ind + 3] = np.zeros(3)
        # if "rot_vel" in info:
        #     self.sim_state.qvel[start_ind + 3:start_ind + 6] = info["rot_vel"]
        # else:
        self.sim_state.qvel[start_ind + 3:start_ind + 6] = np.zeros(3)
        self.sim.set_state(self.sim_state)

    def internal_step(self, action=None):
        ablock = False
        if action is None:
            self.sim.forward()
            self.sim.step()
        else:
            pick_place = action[:3]
            drop_place = action[3:]

            ablock = self.pick_block(pick_place)
            if (ablock):
                # print("Dropping: {} {}".format(ablock, drop_place))
                self.add_block(ablock, drop_place)
        self.sim_state = self.sim.get_state()
        return ablock


    ####Env external step functions
    def step(self, action):
        ablock = self.internal_step(action)
        #if ablock:
        for i in range(self.internal_steps_per_step):
            self.sim.forward()
            self.sim.step()
            if self.view:
                self.viewer.render()

        # self.give_down_vel()
        # for i in range(200):
        #     self.sim.forward()
        #     self.sim.step()
        self.sim_state = self.sim.get_state()
        return self.get_observation()

    def give_down_vel(self):
        for block in self.names:
            self.set_block_info(block, {'vel': [0, 0, 0.5]})

    def check_contact(self, drop_name):
        while contacts.is_overlapping(self.sim, drop_name):
            joint_ind = self.sim.model._joint_name2id[drop_name]
            qpos_start_ind = joint_ind * 7

            self.sim.data.qpos[qpos_start_ind + 2] += 0.05
            self.sim.forward()

    def reset(self):
        self.names = []
        self.blocks = []
        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            poly = np.random.choice(self.polygons)
            pos = self.get_random_pos()
            pos[-2] += 2 * (i + 1)
            self.add_mesh(poly, pos, quat, self.get_random_rbga(self.num_colors))
        self.initialize(False)
        return self.get_observation()

    def get_observation(self):
        img = self.sim.render(self.img_dim, self.img_dim, camera_name="fixed") #img is upside down, values btwn 0-255
        img = img[::-1, :, :]
        return np.ascontiguousarray(img) #values btwn 0-255

    def get_obs_size(self):
        return [self.img_dim, self.img_dim]

    def get_actions_size(self):
        return [6]

    def get_rand_block_byz(self):
        if len(self.names) == 0:
            raise KeyError("No blocks in get_rand_block_byz()!")
        if self.include_z:

            aname = np.random.choice(self.names)
        else:

            def x_y_thresh(pos1, pos2):
                if np.linalg.norm(pos1[:2] - pos2[:2]) < 0.2:
                    return True
                else:
                    return False
            def taller(pos1, pos2):
                if pos1[-1] > pos2[-1]:
                    return True
                else:
                    return False

            z_lim = 0.5
            poss = {aname: self.get_block_info(aname)['pos'] for aname in self.names}
            clusters = [[self.names[0]]]
            for aname in self.names[1:]:
                for i, c in enumerate(clusters):
                    if x_y_thresh(poss[aname], poss[c[-1]]):
                        if taller(poss[aname], poss[c[-1]]):
                            clusters[i].append(aname)
                        break
                    else:
                        clusters.append([aname])

            cluster_idx = np.random.choice(range(len(clusters)))
            aname = clusters[cluster_idx][-1]



            # tmp = [aname for aname in self.names if self.get_block_info(aname)["pos"][2] <= z_lim]
            # while (len(tmp) == 0):
            #     z_lim += 0.5
            #     tmp = [aname for aname in self.names if self.get_block_info(aname)["pos"][2] <= z_lim]
            # aname = np.random.choice(tmp)
        return aname

    def sample_action(self, action_type=None):
        if action_type == 'pick_block': #pick block, place randomly
            # aname = np.random.choice(self.names)
            aname = self.get_rand_block_byz()
            pick = self.get_block_info(aname)["pos"] + np.random.randn(3)/10
            place = self.get_random_pos(self.drop_heights)
        elif action_type == 'place_block': #pick block, place on top of existing block
            # aname = np.random.choice(self.names)
            aname = self.get_rand_block_byz()
            pick = self.get_block_info(aname)["pos"] + np.random.randn(3)/10
            names = copy.deepcopy(self.names)
            names.remove(aname)
            aname = np.random.choice(names)
            place = self.get_block_info(aname)["pos"] + np.random.randn(3)/10
            place[2] = self.drop_heights
        elif action_type == 'remove_block':
            aname = self.get_rand_block_byz()
            pick = self.get_block_info(aname)["pos"] + np.random.randn(3)/50
            place = [0, 0, -5] #Place the block under the ground to remove it from scene
        elif action_type is None:
            if self.include_z:
                pick = self.get_random_pos()
                place = self.get_random_pos(self.drop_heights)
            else:
                pick = self.get_random_pos(0.2)
                place = self.get_random_pos(3.5)
        else:
            raise KeyError("Wrong input action_type!")
        return np.array(list(pick) + list(place))

    def sample_action_gaussian(self, mean, std):
        random_a = np.random.normal(mean, std)
        # set pick height
        random_a[2] = 0.2
        # set place height
        random_a[5] = self.drop_heights

        return random_a

    def get_drop_pos(self, index):
        delta_x = 1
        y_val = 0
        left_most_x = -2.5
        return [left_most_x+index*delta_x, y_val, 4]

    def get_valid_width_pos(self, width):
        num_pos = len(self.heights)
        possible = []
        for i in range(num_pos):
            valid = True
            for k in range(max(i-width, 0), min(i+width+1, num_pos)):
                if self.types[k] == "tetrahedron":
                    valid = False
                    break
                if self.heights[i] < self.heights[k]:
                    valid = False
                    break
                if self.heights[i] >= 3:
                    valid = False
                    break
            if valid:
                possible.append(i)
        return possible

    def create_tower_shape(self):
        self.names = []
        self.blocks = []

        self.heights = [0, 0, 0, 0, 0]
        self.types = [None] * 5
        self.check_clear_width = {'cube' : 1, 'horizontal_rectangle' : 1, 'tetrahedron' : 1}
        self.add_height_width = {'cube' : 0, 'horizontal_rectangle' : 1, 'tetrahedron' : 0}

        tmp_polygons = copy.deepcopy(self.polygons) #['cube', 'horizontal_rectangle', 'tetrahedron'][:2]

        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            poly = np.random.choice(tmp_polygons)
            tmp = self.get_valid_width_pos(self.check_clear_width[poly])
            if len(tmp) == 0:
                tmp_polygons.remove(poly)
                if len(tmp_polygons) == 0:
                    # print("DONE!")
                    break
                else:
                    continue

            tmp_polygons = copy.deepcopy(self.polygons)
            ind = np.random.choice(tmp)
            # print(poly, tmp, ind)
            self.update_tower_info(ind, poly)
            tmp_pos = self.get_drop_pos(ind)
            self.add_mesh(poly, tmp_pos, quat, self.get_random_rbga(self.num_colors))
        self.num_objects = len(self.names)
        self.initialize(True)

    def update_tower_info(self, ind, poly):
        self.types[ind] = poly
        width = self.add_height_width[poly]
        new_height = self.heights[ind] + 1
        for i in range(max(ind-width,0), min(ind+width+1, len(self.heights))):
            # print(i, new_height)
            self.heights[i] = new_height

        for i in range(1,4):
            if self.heights[i-1] == self.heights[i+1] and new_height == self.heights[i-1]:
                self.heights[i] = self.heights[i-1]

        print(poly, ind, self.types, self.heights)

    def get_env_info(self):
        env_info = {}
        env_info["names"] = self.names
        env_info["blocks"] = self.blocks
        return env_info

    def set_env_info(self, env_info):
        self.names = env_info["names"]
        self.blocks = env_info["blocks"]
        self.initialize(True)


def createSingleSim(args):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    num_blocks = np.random.randint(args.min_num_objects, args.max_num_objects+1)
    myenv = BlockPickAndPlaceEnv(num_blocks, args.num_colors, args.img_dim, args.include_z,
                                 random_initialize=True)
    myenv.img_dim = args.img_dim
    imgs = []
    acs = []
    imgs.append(myenv.get_observation())
    rand_float = np.random.uniform()
    for t in range(args.num_frames-1):
        if args.remove_objects == 'True':
            ac = myenv.sample_action('remove_block')
        else:
            if rand_float < args.force_pick:
                ac = myenv.sample_action('pick_block')
            elif rand_float < args.force_pick + args.force_place:
                ac = myenv.sample_action('place_block')
            else:
                ac = myenv.sample_action()
        imgs.append(myenv.step(ac))
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
    parser.add_argument('-fpick', '--force_pick', type=float, default=0.5)
    parser.add_argument('-fplace', '--force_place', type=float, default=0.5)
    parser.add_argument('-r', '--remove_objects', type=bool, default=False)
    parser.add_argument('-z', '--include_z', type=bool, default=False)
    parser.add_argument('--output_path', default='', type=str,
                        help='path to save images')
    parser.add_argument('-p', '--num_workers', type=int, default=1)

    args = parser.parse_args()
    print(args)
    info = {}
    info["min_num_objects"] = args.min_num_objects
    info["max_num_objects"] = args.max_num_objects
    info["img_dim"] = args.img_dim

    if args.remove_objects:
        args.num_frames = 2

    info["num_frames"] = args.num_frames
    # single_sim_func = lambda : createSingleSim(args)
    single_sim_func = createSingleSim
    env = BlockPickAndPlaceEnv(1, 1, args.img_dim, args.include_z, random_initialize=True)
    ac_size = env.get_actions_size()
    obs_size = env.get_obs_size()
    dgu.createMultipleSims(args, obs_size, ac_size, single_sim_func, num_workers=int(args.num_workers))

    dgu.hdf5_to_image(args.filename)
    for i in range(min(10, args.num_sims)):
        tmp = os.path.join(args.output_path, "imgs/training/{}/features".format(str(i)))
        dgu.make_gif(tmp, "animation.gif")
        # tmp = os.path.join(args.output_path, "imgs/training/{}/groups".format(str(i)))
        # dgu.make_gif(tmp, "animation.gif")

    # cur_fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(4 * 6, 1 * 6))
    # b = BlockPickAndPlaceEnv(6, None, 64, include_z=False, random_initialize=False, view=False)
    # for i in range(5):
    #     if i == 0:
    #         b.create_tower_shape()
    #     else:
    #         b.step(b.sample_action("pick_block"))
    #     ob = b.get_observation()
    #     axes[i].imshow(ob, interpolation='nearest')
    # cur_fig.savefig("HELLO")




    # b = BlockPickAndPlaceEnv(6, None, 64, False, random_initialize=False, view=True)
    # b.create_tower_shape()
    # ob = b.get_observation()
    #
    # env_info = b.get_env_info()
    #
    # c = BlockPickAndPlaceEnv(6, None, 64, False, random_initialize=False, view=False)
    # c.set_env_info(env_info)
    # b=c
    # for i in range(10):
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
    #         # print(i)
    #         # for aname in b.names:
    #         #     print(b.get_block_info(aname))
    #     # b.viewer.render()



    #Running and rendering example
    # b = BlockPickAndPlaceEnv(4, None, 64, True, True, view=True)
    # # b = BlockPickAndPlaceEnv(4, None, 64, include_z, random_initialize=False, view=False)
    # for i in range(10):
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
    #         # print(i)
    #         # for aname in b.names:
    #         #     print(b.get_block_info(aname))
    #     b.viewer.render()
    #
    # #Plotting images
    # cur_fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(4 * 6, 1 * 6))
    #
    # myenv = BlockPickAndPlaceEnv(4, None, 64, include_z=False)
    # myenv.create_tower_shape()
    # obs = []
    # obs.append(myenv.get_observation())
    # axes[0].imshow(obs[-1], interpolation='nearest')
    # cur_fig.savefig("HELLO")
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
    #     ac = myenv.sample_action("pick_block")
    #     tmp = myenv.step(ac)
    #     print(tmp.shape, np.max(tmp), np.min(tmp), np.mean(tmp))
    #     # print(np.max(tmp), np.min(tmp), np.mean(tmp))
    #     axes[i+1].set_title(ac)
    #     axes[i + 1].imshow(tmp)#, interpolation='nearest')
    #     # for k in range(5):
    #     #     axes[i+1, k].imshow(tmp[k], interpolation='nearest')
    # cur_fig.savefig("HELLO")
