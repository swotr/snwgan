import numpy as np
import tensorflow as tf

def conv(l, filters, kernel, stride, name):
    return tf.layers.conv2d(l, filters, kernel, strides=stride, padding='same',
        kernel_initializer=tf.glorot_uniform_initializer(np.sqrt(2)), name=name)

def deconv(l, filters, kernel, stride, name):
    return tf.layers.conv2d_transpose(l, filters, kernel, strides=stride, padding='same', 
        kernel_initializer=tf.glorot_uniform_initializer(np.sqrt(2)), name=name)

def dense(l, filters, name):
    return tf.layers.dense(l, filters,
        kernel_initializer=tf.glorot_uniform_initializer(np.sqrt(2)), name=name)

def spectral_normed_weight(w, num_iters=1, update_collection=None):
    """Performs Spectral Normalization on a weight tensor.

    Specifically it divides the weight tensor by its largest singular value. This
    is intended to stabilize GAN training, by making the discriminator satisfy a
    local 1-Lipschitz constraint.
    Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
    [sn-gan] https://openreview.net/pdf?id=B1QRgziT-

    Args:
    w: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
                        If None, the function will update u during the forward
                        pass. Else if the update_collection equals 'NO_OPS', the
                        function will not update the u during the forward. This
                        is useful for the discriminator, since it does not update
                        u in the second pass.
                        Else, it will put the assignment in a collection
                        defined by the user. Then the user need to run the
                        assignment explicitly.
    """    
    w_shape = w.shape.as_list()
    w_r = tf.reshape(w, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(), trainable=False)
    _u = u
    _v = None
    for _ in range(num_iters):
        _v = tf.nn.l2_normalize(tf.matmul(_u, w_r, transpose_b=True))
        _u = tf.nn.l2_normalize(tf.matmul(_v, w_r))
    
    sigma = tf.squeeze(tf.matmul(tf.matmul(_v, w_r), _u, transpose_b=True))
    tf.summary.scalar('sigma', sigma) # DEBUG
    w_r /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(_u)]):
            w_r = tf.reshape(w_r, w_shape)
    else:
        w_r = tf.reshape(w_r, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(_u))
    return w_r

def snconv2d(name, input, output_dim, kernel_size, stride, seed=None, update_collection=None):
    ''' spectral normed convolution '''
    with tf.variable_scope(name):
        w = tf.get_variable('W', [kernel_size, kernel_size, input.get_shape()[-1], output_dim], tf.float32, 
            initializer=tf.glorot_uniform_initializer(seed), trainable=True)        
        w_bar = spectral_normed_weight(w, update_collection=update_collection)
        conv = tf.nn.conv2d(input, w_bar, strides=[1, stride, stride, 1], padding='SAME')
        biases = tf.get_variable('b', [output_dim], 
            initializer=tf.constant_initializer(0.0), trainable=True)
        conv = tf.nn.bias_add(conv, biases)
        return conv

def snlinear(name, input, output_size, seed=None, use_bias=True, update_collection=None):
    ''' spectral normed linear '''
    shape = input.shape.as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('W', [shape[1], output_size], tf.float32,
            initializer=tf.glorot_uniform_initializer(seed), trainable=True)        
        w_bar = spectral_normed_weight(w, update_collection=update_collection)
        feat = tf.matmul(input, w_bar)
        if use_bias:
            bias = tf.get_variable('b', [output_size], 
                initializer=tf.constant_initializer(0.0), trainable=True)
            feat = tf.nn.bias_add(feat, bias)
        return feat

def snembed(name, input, output_size, seed=None, update_collection=None):
    ''' spectral normed word embed
        input: tensor of one-hot (labels) [batch_size, NUM_CLASSES]
        output_size: feature dimension of word-embedded vector '''
    shape = input.shape.as_list()
    with tf.variable_scope(name):        
        w = tf.get_variable('W', [shape[1], output_size], tf.float32,
            initializer=tf.glorot_uniform_initializer(seed), trainable=True)
        w_bar = spectral_normed_weight(w, update_collection=update_collection)
        #feat = tf.gather(w_bar, tf.cast(tf.squeeze(input), tf.int32)) # scalar
        feat = tf.matmul(input, w_bar) # one-hot
        return feat