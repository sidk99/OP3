import os
import pdb
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

MODEL_XML_BASE = """
<?xml version="1.0" ?>
<mujoco>
    <asset>
       {}
       {}
    </asset>
    <worldbody>
        <body name="floor" pos="0 0 0.025">
            <geom size="3.0 3.0 0.02" rgba="0 1 0 1" type="box"/>
            <camera name='fixed' pos='0 -8 8' euler='45 0 0'/>
        </body>
        {}
    </worldbody>
    <option gravity="0 0 -9.81"/>
</mujoco>
"""

#Red, Lime, Blue, Yellow, Cyan, Magenta, Black, White
COLOR_LIST = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [1, 1, 1], [255, 255, 255]]
def pickRandomColor(an_int):
    if an_int is None:
        return np.random.uniform(low=0.0, high=1.0, size=3)
    tmp = np.random.randint(0, an_int)
    return np.array(COLOR_LIST[tmp])/255


class BlockPickAndPlaceEnv():
    def __init__(self, num_objects, num_colors):
        self.asset_path = os.path.join(os.getcwd(), '../data/stl/')
        self.img_dim = 64
        self.polygons = ['cube', 'horizontal_rectangle', 'tetrahedron']
        self.num_colors = num_colors
        self.internal_steps_per_step = 1000

        self.names = []
        self.blocks = []

        poly = 'cube'
        quat = [1, 0, 0, 0]
        for i in range(num_objects):
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
        asset_base = '<mesh name="{}" scale="0.4 0.4 0.4" file="{}"/>'
        asset_list = [asset_base.format(a['name'], os.path.join(self.asset_path, a['polygon'] + '.stl'))
                      for a in self.blocks]
        asset_str = '\n'.join(asset_list)
        return asset_str

    def get_body_str(self):
        body_base = '''
          <body name='{}' pos='{}' quat='{}'>
            <joint type='free' name='{}' damping="1"/>
            <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}' condim="6" friction="1 1 1"/>
          </body>
        '''
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
        if height:
            tmp = list(np.random.randn(2)) + list(np.abs(np.random.rand(1))+0.2)
            return tmp
        tmp = list(np.random.randn(2)) #Get random x,y position
        return tmp + [height]

    def get_random_rbga(self, num_colors):
        rgb = list(pickRandomColor(num_colors))
        return rgb + [1]

    def initialize(self):
        tmp = MODEL_XML_BASE.format(self.get_asset_mesh_str(), self.get_asset_material_str(), self.get_body_str())
        model = load_model_from_xml(tmp)
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)
        self.sim_state = self.sim.get_state()
        self.get_starting_step()

    def get_starting_step(self):
        for aname in self.names:
            self.add_block(aname, [-5, -5, -5])

        for aname in self.names:
            tmp_pos = self.get_random_pos(2)
            self.add_block(aname, tmp_pos)
            for i in range(2000):
                self.single_step()
                self.viewer.render()
            self.sim_state = self.sim.get_state()


    ####Env internal step functions
    def add_block(self, ablock, pos):
        #pos (x,y,z)
        self.set_block_info(ablock, {"pos": pos})
        self.sim.set_state(self.sim_state)

    def pick_block(self, pos):
        block_name = None
        for a_block in self.names:
            if self.intersect(a_block, pos):
                block_name = a_block

        if block_name is None:
            return False

        PICK_LOC = np.array([0, 0, 2])
        info = {"pos":PICK_LOC}
        self.set_block_info(block_name, info)
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

    def single_step(self, action=None):
        ablock = False
        if action is None:
            self.sim.step()
        else:
            pick_place = action[:3]
            drop_place = action[3:]

            ablock = self.pick_block(pick_place)
            if (ablock):
                # print("Dropping: {}".format(ablock))
                self.add_block(ablock, drop_place)
        self.sim_state = self.sim.get_state()
        return ablock


    ####Env external step functions
    def step(self, action):
        ablock = self.single_step(action)
        if ablock:
            for i in range(self.internal_steps_per_step):
                self.single_step()

        img = self.get_obs()
        return img

    def reset(self):
        self.names = []
        self.blocks = []
        poly = 'cube'
        quat = [1, 0, 0, 0]
        for i in range(self.num_objects):
            self.add_mesh(poly, self.get_random_pos(), quat, self.get_random_rbga(self.num_colors))
        self.initialize()

    def get_obs(self):
        img = self.sim.render(self.img_dim, self.img_dim, camera_name="fixed")
        return img

    def sample_action(self):
        pick = self.get_random_pos(0.2)
        place = self.get_random_pos(3)
        return np.array(list(pick) + list(place))



if __name__ == '__main__':
    b = BlockPickAndPlaceEnv(4, None)
    for i in range(10000):
        # pdb.set_trace()
        b.single_step()
        # if i == 2000:
        #     cur_pos = b.get_block_info("cube_0")["pos"]
        #     ac = list(cur_pos) + [-0.5, -0.5, 1]
        #     b.step(ac)
        if i < 2000:
            # cur_pos = b.get_block_info("cube_0")["pos"]
            # ac = list(cur_pos) + [-0.5, -0.5, 1]
            b.single_step(b.sample_action())

        b.viewer.render()