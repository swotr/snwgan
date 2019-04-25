import numpy as np
import tensorflow as tf

from snops import *

def upscale(x):
    return tf.image.resize_nearest_neighbor(x, 
        [x.shape.as_list()[1]*2, x.shape.as_list()[2]*2])

def upscale_bilinear(x):
    return tf.image.resize_bilinear(x, 
        [x.shape.as_list()[1]*2, x.shape.as_list()[2]*2])

def downscale(x):
    return tf.layers.average_pooling2d(x, 2, 2)

def batch_norm(x, is_training):
    ''' in-place update version'''
    return tf.contrib.layers.batch_norm(x, decay=0.99, center=True, scale=True, epsilon=2e-5,
        updates_collections=None, is_training=is_training) # setting updates_collections=None to force in-place update (refer to API guide)

def resnet_block(name, x, filters, kernel_size, scale, is_training):
    ''' 
    preactivation
           |-BN-->ReLU-->C-->BN-->ReLU-->C-|
      x_l----------------------------------add-->x_(l+1)

    '''
    with tf.variable_scope(name):        
        l = x

        l = batch_norm(l, is_training)
        l = tf.nn.relu(l)

        if scale == 'up':
            l = upscale(l)
        l = tf.layers.conv2d(l, filters, kernel_size, padding='same',
            kernel_initializer=tf.glorot_uniform_initializer(np.sqrt(2)), name='conv1')

        l = batch_norm(l, is_training)
        l = tf.nn.relu(l)

        l = tf.layers.conv2d(l, filters, kernel_size, padding='same',
            kernel_initializer=tf.glorot_uniform_initializer(np.sqrt(2)), name='conv2')
        if scale == 'down':
            l = downscale(l)

        sc = x

        if scale == 'up':
            sc = upscale(sc)
        elif scale == 'down':
            sc = downscale(sc)
        if sc.shape.as_list()[-1] != filters or scale == 'up' or scale == 'down':
            sc = tf.layers.conv2d(sc, filters, 1, padding='same',
                kernel_initializer=tf.glorot_uniform_initializer(np.sqrt(2)), name='sc_conv')

    return l + sc

def snresnet_block_D1(name, x, filters, kernel_size, scale, update_collection):
    '''
    almost same with sn_resnet_block except differnt normalization and non-linearity scheme
    usually used for the first layer of discriminator
    '''
    with tf.variable_scope(name):
        l = x        
        l = snconv2d('sn_conv1', l, filters, kernel_size, 1,
            seed=np.sqrt(2), update_collection=update_collection)
        l = tf.nn.relu(l)
        l = snconv2d('sn_conv2', l, filters, kernel_size, 1, 
            seed=np.sqrt(2), update_collection=update_collection)
        if scale == 'down':
            l = downscale(l)        

        sc = x
        if scale == 'down':
            sc = downscale(sc)
        if sc.shape.as_list()[-1] != filters or scale == 'down':
            sc = snconv2d('sn_conv_sc', sc, filters, 1, 1, 
                seed=np.sqrt(2), update_collection=update_collection)
        
    return l + sc

def snresnet_block(name, x, filters, kernel_size, scale, update_collection, is_training=True):
    ''' 
    w/ spectral normalization    
    preactivation   
           |-BN-->ReLU-->C-->BN-->ReLU-->C-|
      x_l----------------------------------add-->x_(l+1)

    '''
    with tf.variable_scope(name):
        l = x
        #l = batch_norm(l, is_training)
        l = tf.nn.relu(l)
        if scale == 'up':
            l = upscale(l)
        l = snconv2d('sn_conv1', l, filters, kernel_size, 1, 
            seed=np.sqrt(2), update_collection=update_collection)
        #l = batch_norm(l, is_training)
        l = tf.nn.relu(l)
        l = snconv2d('sn_conv2', l, filters, kernel_size, 1, 
            seed=np.sqrt(2), update_collection=update_collection)
        if scale == 'down':
            l = downscale(l)

        sc = x
        if scale == 'up':
            sc = upscale(sc)
        elif scale == 'down':
            sc = downscale(sc)
        if sc.shape.as_list()[-1] != filters or scale == 'up' or scale == 'down':
            sc = snconv2d('sn_conv_sc', sc, filters, 1, 1, 
                seed=np.sqrt(2), update_collection=update_collection)

    return l + sc
