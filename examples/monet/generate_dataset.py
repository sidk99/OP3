import argparse, os, json
import h5py
import numpy as np
from scipy.misc import imread, imresize
import torch

'''python examples/monet/generate_dataset.py --input_image_dir ~/Downloads/CLEVR_v1.0/images/train/ --max_images 10000 --output_h5_file ~/objects/rlkit
/examples/monet/clevr_train.hdf5'''

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)

parser.add_argument('--image_height', default=84, type=int)
parser.add_argument('--image_width', default=84, type=int)

def main(args):
    input_paths = []
    idx_set = set()
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith('.png'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))
        idx_set.add(idx)
    input_paths.sort(key=lambda x: x[1])

    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1

    if args.max_images is not None:
        input_paths = input_paths[:args.max_images]
    print(input_paths[0])
    print(input_paths[-1])

    img_size = (args.image_height, args.image_width)
    with h5py.File(args.output_h5_file, 'w') as f:
        N = len(input_paths)
        C = 3
        H = args.image_height
        W = args.image_width
        feat_dset = f.create_dataset('features', (N, C, H, W),
                                                 dtype=np.uint8)
        for i, (path, idx) in enumerate(input_paths):
            img = imread(path, mode='RGB')
            img = img[29:221, 64:256, :]

            img = imresize(img, img_size, interp='bicubic')
            img = img.transpose(2, 0, 1)[None]
            feat_dset[i] = img

                #print('Processed %d / %d images' % (i0, len(input_paths)))

if __name__ == '__main__':
    main(parser.parse_args())