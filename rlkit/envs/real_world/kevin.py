import argparse
import cv2
import glob
import h5py
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

import dataset_utils as du

def decode_and_sample(serialized_example):
	########################################
	# Constants for Kevin's dataset
	sampling_max_horizon = 14
	img_h = 100
	img_w = 100
	img_c = 3
	act_dim = 2
	joint_dim = 3
	########################################

	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
			'images': tf.FixedLenFeature([], tf.string),
			'actions': tf.FixedLenFeature([sampling_max_horizon*(act_dim+1)], tf.float32),
			'qts': tf.FixedLenFeature([sampling_max_horizon*joint_dim], tf.float32),
			})

	# Decode and normalize images.
	images = tf.decode_raw(features['images'], tf.uint8)
	images.set_shape(((sampling_max_horizon+1) * img_h * img_w * img_c,))
	images = tf.reshape(images, [sampling_max_horizon+1, img_h, img_w, img_c])

	# Decode actions and states.
	actions = tf.cast(features['actions'], tf.float32)
	actions = tf.reshape(actions, [sampling_max_horizon, (act_dim+1)])
	actions = actions[:, :-1] # z velocity is always 0
	qt = tf.cast(features['qts'], tf.float32)
	qt = tf.reshape(qt, [sampling_max_horizon, joint_dim])

	return images, actions, qt

def get_iterators(args):
	datasets = {
		'train': glob.glob(os.path.join(args.root, args.train_path)), 
		'val': glob.glob(os.path.join(args.root, args.val_path))}
	iterators = {}
	for mode in datasets:
		dataset = tf.data.TFRecordDataset(datasets[mode])
		dataset = dataset.repeat(count=1)  # one pass through the epoch
		dataset = dataset.map(lambda example: decode_and_sample(example), num_parallel_calls=10)
		dataset = dataset.batch(1)
		print('Processed dataset')
		iterators[mode] = dataset.make_one_shot_iterator()
	return iterators

def read_kevin(args):
	iterators = get_iterators(args)

	sess = tf.Session()
	num_sims = {'train': 0, 'test': 0, 'val': 0}
	hps = du.HyperParams(['num_frames', 'image_res', 'action_dim'])

	for mode in ['train', 'val']:
		iterator = iterators[mode]
		images_tensor, actions_tensor, qt_tensor = iterator.get_next()
		counter = 0
		while True:
			try:
				obs_np = np.squeeze(sess.run(images_tensor), axis=0)  # (15, 100, 100, 3)
				act_np = np.squeeze(sess.run(actions_tensor), axis=0)  # (14, 2)
				qt_np = np.squeeze(sess.run(qt_tensor), axis=0)  # (14, 3)

				hps.update_hp('num_frames', obs_np.shape[0])
				hps.update_hp('image_res', 64)  # manually impose. Will resize in write_all_data
				hps.update_hp('action_dim', act_np.shape[-1])

				counter += 1
			except:
				break
		print('{} examples in mode {}'.format(counter, mode))
		num_sims[mode] += counter

	mode_map = {'training': 'train', 'validation': 'val'}
	return num_sims, hps, iterators, mode_map

def write_kevin(iterators, mode, num_sims, num_frames, features_dataset, action_dataset):
    sess = tf.Session()
    iterator = iterators[mode]
    images_tensor, actions_tensor, qt_tensor = iterator.get_next()
    for i in tqdm(range(num_sims)):
        obs_np = np.squeeze(sess.run(images_tensor), axis=0)  # (15, 100, 100, 3)
        act_np = np.squeeze(sess.run(actions_tensor), axis=0)  # (14, 2)
        qt_np = np.squeeze(sess.run(qt_tensor), axis=0)  # (14, 3)

        obs_np = np.stack([cv2.resize(obs_np[j], dsize=(64, 64), interpolation=cv2.INTER_CUBIC) for j in range(num_frames)])

        features_dataset[:, i, :, :, :] = obs_np  # (num_frames, 64, 64, 3)
        action_dataset[:, i, :] = act_np  # (num_frames-1, 2)

