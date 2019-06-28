import os
import h5py
import cv2
import numpy as np
import imageio
from progressbar import ProgressBar

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class HyperParams():
    def __init__(self, keys):
        self.hp = {k: None for k in keys}

    def update_hp(self, hp_name, value):
        if hp_name not in self.hp:
            assert False
        else:
            if self.hp[hp_name] is None:
                self.hp[hp_name] = value
            else:
                old_value = self.hp[hp_name]
                assert old_value == value

    def get(self, hp_name):
        return self.hp[hp_name]

def make_gif(images_root, gifname):
    file_names = [fn for fn in os.listdir(images_root) if fn.endswith('.png')]
    file_names =  sorted(file_names, key=lambda x: int(os.path.splitext(x)[0]))
    images = []
    for a_file in file_names:
        images.append(imageio.imread(os.path.join(images_root,a_file)))
    imageio.mimsave(images_root + gifname, images)

def mkdirp(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder

def hdf5_to_image(filename, num_examples):
    root = os.path.dirname(filename)
    img_root = mkdirp(os.path.join(root, 'imgs'))
    h5file = h5py.File(filename, 'r')
    for mode in h5file.keys():
        mode_folder = mkdirp(os.path.join(img_root, mode))
        groups = h5file[mode]
        f = groups['features']
        for ex in range(f.shape[1])[:num_examples]:
            ex_folder = mkdirp(os.path.join(mode_folder, str(ex)))
            for d in groups.keys():
                if d in ['features']:
                    dataset_folder = mkdirp(os.path.join(ex_folder, d))
                    dataset = groups[d]
                    num_groups = np.max(dataset[:, ex])
                    for j in range(dataset.shape[0]):
                        imfile = os.path.join(dataset_folder, str(j)+'.png')
                        if d == 'features':
                            plt.imsave(imfile, dataset[j, ex])
                        else:
                            assert False
