#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

# Paths
image_path = '/home/minje/dev/dataset/cifar/cifar-10-fake' # set path to some generated images
#image_path = '/home/minje/dev/dataset/stl/fake-images' # set path to some generated images
stats_path = '/home/minje/dev/dataset/cifar/fid_stats_cifar10.npz' # training set statistics (maybe pre-calculated)
#stats_path = '/home/minje/dev/dataset/stl/fid_stats_stl10.npz' # training set statistics (maybe pre-calculated)
inception_path = fid.check_or_download_inception(None) # download inception network

# precalculate training set statistics
# #image_files = glob.glob(os.path.join('/home/minje/dev/dataset/cifar/cifar-10-images', '*.jpg'))
# image_files = glob.glob(os.path.join('/home/minje/dev/dataset/stl/images', '*.jpg'))
# fid.create_inception_graph(inception_path)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     mu_real, sigma_real = fid.calculate_activation_statistics_from_files(image_files, sess,
#         batch_size=100, verbose=True)
# np.savez(stats_path, mu=mu_real, sigma=sigma_real)
# exit(0)

# loads all images into memory (this might require a lot of RAM!)
image_files = glob.glob(os.path.join(image_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_files])

# load precalculated training set statistics
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100, verbose=True)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)
