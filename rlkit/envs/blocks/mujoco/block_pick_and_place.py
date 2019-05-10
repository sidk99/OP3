import os
import pdb
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import rlkit.envs.blocks.mujoco.utils.data_generation_utils as dgu

from mujoco_py import load_model_from_xml, MjSim, MjViewer

MODEL_XML_BASE = """
<mujoco>
    <asset>
        <material name='wall_visible' rgba='.9 .9 .9 1' specular="0" shininess="0"  emission="0.25"/>
        <material name='wall_invisible' rgba='.9 .9 .9 0' specular="0" shininess="0" emission="0.25"/>
       {}
       {}
    </asset>
    <worldbody>
        <camera name='fixed' pos='0 -8 6' euler='-300 0 0'/>
        <geom name='wall_left'  type='box' pos='-5 0 0' euler='0 0 0' size='0.1 10 4' material='wall_visible'/>
        <geom name='wall_right'  type='box' pos='5 0 0' euler='0 0 0' size='0.1 10 4' material='wall_visible'/>
        <geom name='wall_back'  type='box' pos='0 5 0' euler='0 0 0' size='10 0.1 4' material='wall_visible'/>
        
        <geom name='wall_floor' type='plane' pos='0 0 0' euler='0 0 0' size='20 10 0.1' material='wall_visible' condim="6" friction="1 1 1"/>
        {}
    </worldbody>
    <option gravity="0 0 -30"/>
</mujoco>
"""
#60 = -300
# <geom name='wall_front'  type='box' pos='0 -5 0' euler='0 0 0' size='10 0.1 4' material='wall_visible'/>

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


class BlockPickAndPlaceEnv():
    def __init__(self, num_objects, num_colors, img_dim):
        self.asset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/stl/')
        self.img_dim = img_dim
        self.polygons = ['cube', 'horizontal_rectangle', 'tetrahedron']
        self.num_colors = num_colors
        self.num_objects = num_objects
        self.internal_steps_per_step = 1000
        self.bounds = {'x_min':-4, 'x_max':4, 'y_min':-1.5, 'y_max':3, 'z_min':0.05, 'z_max': 2.2}

        self.names = []
        self.blocks = []

        poly = 'cube'
        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            self.add_mesh(poly, self.get_random_pos(), quat, self.get_random_rbga(num_colors))
        self.initialize()

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
        self.blocks.append({'name': name, 'polygon': polygon, 'pos': pos, 'quat': quat, 'rgba': rgba,
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
            <joint type='free' name='{}' damping="1"/>
            <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}' condim="6" friction="1 0.005 0.0001"/>
          </body>
        '''
        # '''<geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}' condim="6" friction="1 1 1"/>'''
        body_list = [body_base.format(m['name'], self.convert_to_str(m['pos']), self.convert_to_str(m['quat']), m['name'],
                                      m['name'], m['name'], m['material'])
            for m in self.blocks]
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
            z = np.random.uniform(self.bounds['z_min'], self.bounds['z_max'])
        else:
            z = height
        return np.array([x, y, z])

    def get_random_rbga(self, num_colors):
        rgb = list(pickRandomColor(num_colors))
        return rgb + [1]

    def initialize(self):
        tmp = MODEL_XML_BASE.format(self.get_asset_mesh_str(), self.get_asset_material_str(), self.get_body_str())
        model = load_model_from_xml(tmp)
        self.sim = MjSim(model)
        # self.viewer = MjViewer(self.sim)
        self.sim_state = self.sim.get_state()
        self.get_starting_step()

    def get_starting_step(self):
        for aname in self.names:
            self.add_block(aname, [-5, -5, -5])

        for aname in self.names:
            tmp_pos = self.get_random_pos(2)
            self.add_block(aname, tmp_pos)
            for i in range(self.internal_steps_per_step):
                self.internal_step()
                # self.viewer.render()
            self.sim_state = self.sim.get_state()


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

        PICK_LOC = np.array([0, 0, 5])
        info = {"pos":PICK_LOC}
        self.set_block_info(block_name, info)
        # self.sim.set_state(self.sim_state)
        return block_name

    def intersect(self, a_block, pos):
        #Threshold
        THRESHHOLD = 0.2
        cur_pos = self.get_block_info(a_block)["pos"]
        return np.max(np.abs(cur_pos - pos)) < THRESHHOLD

    def get_block_info(self, a_block):
        info = {}
        info["pos"] = self.sim.data.get_body_xpos(a_block) #np array
        info["quat"] = self.sim.data.get_body_xquat(a_block)
        info["vel"] = self.sim.data.get_body_xvelp(a_block)
        info["rot_vel"] = self.sim.data.get_body_xvelr(a_block)
        return info

    def set_block_info(self, a_block, info):
        # print("Setting state: {}, {}".format(a_block, info))
        start_ind = self.sim.model.get_joint_qpos_addr(a_block)[0]
        if "pos" in info:
            self.sim_state.qpos[start_ind:start_ind+3] = info["pos"]
        if "quat" in info:
            self.sim_state.qpos[start_ind+3:start_ind+7] = info["quat"]
        else:
            self.sim_state.qpos[start_ind + 3:start_ind + 7] = np.zeros(4)

        start_ind = self.sim.model.get_joint_qvel_addr(a_block)[0]
        if "vel" in info:
            self.sim_state.qvel[start_ind:start_ind + 3] = info["vel"]
        else:
            self.sim_state.qvel[start_ind:start_ind + 3] = np.zeros(3)
        if "rot_vel" in info:
            self.sim_state.qvel[start_ind + 3:start_ind + 6] = info["rot_vel"]
        else:
            self.sim_state.qvel[start_ind + 3:start_ind + 6] = np.zeros(3)
        self.sim.set_state(self.sim_state)

    def internal_step(self, action=None):
        ablock = False
        if action is None:
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
        if ablock:
            for i in range(self.internal_steps_per_step):
                self.internal_step()
                # self.viewer.render()
        return self.get_observation()

    def reset(self):
        self.names = []
        self.blocks = []
        poly = 'cube'
        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            self.add_mesh(poly, self.get_random_pos(), quat, self.get_random_rbga(self.num_colors))
        self.initialize()

    def get_observation(self):
        img = self.sim.render(self.img_dim, self.img_dim, camera_name="fixed") #img is upside down, values btwn 0-255
        img = img[::-1, :, :]
        return img/255 #values btwn 0-255

    def get_obs_size(self):
        return [self.img_dim, self.img_dim]

    def get_actions_size(self):
        return [6]

    def sample_action(self, action_type=None):
        if action_type == 'pick_block': #pick block, place randomly
            aname = np.random.choice(self.names)
            place = self.get_random_pos(3.5)
            pick = self.get_block_info(aname)["pos"] + np.random.randn(3)/10
        elif action_type == 'place_block': #pick block, place on top of existing block
            aname = np.random.choice(self.names)
            pick = self.get_block_info(aname)["pos"] + np.random.randn(3)/10
            aname = np.random.choice(self.names)
            place = self.get_block_info(aname)["pos"] + np.random.randn(3)/10
            place[2] = 3.5
        elif action_type is None:
            pick = self.get_random_pos(0.2)
            place = self.get_random_pos(3.5)
        else:
            raise KeyError("Wrong input action_type!")
        return np.array(list(pick) + list(place))


def createSingleSim(args):
    num_blocks = np.random.randint(args.min_num_objects, args.max_num_objects+1)
    myenv = BlockPickAndPlaceEnv(num_blocks, args.num_colors, args.img_dim)
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

    args = parser.parse_args()
    print(args)

    info = {}
    info["min_num_objects"] = args.min_num_objects
    info["max_num_objects"] = args.max_num_objects
    info["img_dim"] = args.img_dim
    info["num_frames"] = args
    single_sim_func = lambda : createSingleSim(args)
    env = BlockPickAndPlaceEnv(1, 1, args.img_dim)
    ac_size = env.get_actions_size()
    obs_size = env.get_obs_size()
    dgu.createMultipleSims(args, obs_size, ac_size, single_sim_func)

    dgu.hdf5_to_image(args.filename)
    for i in range(10):
        tmp = os.path.join(args.output_path, "imgs/training/{}/features".format(str(i)))
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
