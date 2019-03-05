import numpy as np
import tensorflow as tf

def spectral_norm(name, w, iteration=1):
    '''
    SN-GAN
    '''
    with tf.variable_scope(name):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.get_variable('u', [1, w_shape[-1]], 
            initializer=tf.truncated_normal_initializer(), trainable=False) # reuse u but not-trainable

        u_hat = u
        v_hat = None
        for i in range(iteration):
            '''
            the principle of the power method
            usually iteration=1 is enough since we reuse u thru training
            '''
            v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w)))
            u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w))

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))[0, 0]    
        tf.summary.scalar('sigma', sigma) # DEBUG dump        
        with tf.control_dependencies([u.assign(u_hat)]): # reuse u                
            w = w/sigma
            w = tf.reshape(w, w_shape)
            #return w, sigma
            return w

# def sn_conv2d(name, input, filters, kernel_size, stride):    
#     fan_in = kernel_size*kernel_size*input.shape.as_list()[-1]
#     #stddev = np.sqrt(2.0/(fan_in))
#     stddev = 0.02
#     with tf.variable_scope(name):
#         w = tf.get_variable('W', [kernel_size, kernel_size, input.shape.as_list()[-1], filters],
#             initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
#         b = tf.get_variable('b', [filters], initializer=tf.constant_initializer(0.0), trainable=True)
#         w_sn, sigma = spectral_norm(w)        
#         tf.summary.scalar('sigma', tf.reduce_mean(sigma))
#         #w_sn, _ = spectral_norm(w)        
#         l = tf.nn.conv2d(input, filter=w_sn, strides=[1, stride, stride, 1], padding='SAME')
#         l = tf.nn.bias_add(l, b)

#     return l

# def sn_conv2d_transpose(name, input, filters, kernel_size, stride, output_shape):
#     fan_in = kernel_size*kernel_size*input.shape.as_list()[-1]
#     #stddev = np.sqrt(2.0/(fan_in))
#     stddev = 0.02
#     with tf.variable_scope(name):
#         input_shape = input.shape.as_list()
#         w = tf.get_variable('W', [kernel_size, kernel_size, input_shape[-1], filters],
#             initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
#         b = tf.get_variable('b', [filters], initializer=tf.constant_initializer(0.0), trainable=True)        
#         #w_sn, sigma = spectral_norm(w)        
#         #tf.summary.scalar('sigma', tf.reduce_mean(sigma))
#         w_sn, _ = spectral_norm(w)
#         l = tf.nn.conv2d_transpose(input, filter=w_sn, output_shape=output_shape, 
#                 strides=[1, stride, stride, 1], padding='SAME')
#         l = tf.nn.bias_add(l, b)

#     return l

# def sn_fully_connected(name, input, filters):
#     #stddev = np.sqrt(1.0/input.shape.as_list()[-1])
#     stddev = 0.02
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         w = tf.get_variable('W', [input.shape.as_list()[-1], filters],
#             initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
#         b = tf.get_variable('b', [filters], initializer=tf.constant_initializer(0.0), trainable=True)        
#         #w_sn, sigma = spectral_norm(w) 
#         #tf.summary.scalar('sigma', tf.reduce_mean(sigma))       
#         w_sn, _ = spectral_norm(w)
#         l = tf.matmul(input, w_sn)
#         l = tf.nn.bias_add(l, b)

#     return l

def spectral_normed_weight(name, weights, num_iters=1, update_collection=None,
                           with_sigma=False):
    """Performs Spectral Normalization on a weight tensor.

    Specifically it divides the weight tensor by its largest singular value. This
    is intended to stabilize GAN training, by making the discriminator satisfy a
    local 1-Lipschitz constraint.
    Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
    [sn-gan] https://openreview.net/pdf?id=B1QRgziT-

    Args:
    weights: The weight tensor which requires spectral normalization
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
    with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
    Returns:
    w_bar: The normalized weight tensor
    sigma: The estimated singular value for the weight tensor.
    """
    with tf.variable_scope(name):
        w_shape = weights.shape.as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
        u = tf.get_variable('u', [1, w_shape[-1]],
                            initializer=tf.truncated_normal_initializer(),
                            trainable=False)
        u_ = u
        for _ in range(num_iters):
            v_ = tf.nn.l2_normalize(tf.matmul(u_, w_mat, transpose_b=True))
            u_ = tf.nn.l2_normalize(tf.matmul(v_, w_mat))

        v_ = tf.stop_gradient(v_)
        u_ = tf.stop_gradient(u_)

        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
        tf.summary.scalar('sigma', sigma) # DEBUG
        w_mat /= sigma
        if update_collection is None:
            with tf.control_dependencies([u.assign(u_)]):
                w_bar = tf.reshape(w_mat, w_shape)
        else:
            w_bar = tf.reshape(w_mat, w_shape)
            if update_collection != 'NO_OPS':
                tf.add_to_collection(update_collection, u.assign(u_))
        if with_sigma:
            return w_bar, sigma
        else:
            return w_bar

def snconv2d(name, input_, output_dim, kernel_size, stride, seed=None, update_collection=None):
    with tf.variable_scope(name):
        w = tf.get_variable('W', [kernel_size, kernel_size, input_.get_shape()[-1], output_dim], tf.float32, 
            initializer=tf.glorot_uniform_initializer(seed), trainable=True)        
        w_bar = spectral_normed_weight('sp_norm', w, update_collection=update_collection)
        conv = tf.nn.conv2d(input_, w_bar, strides=[1, stride, stride, 1], padding='SAME')
        biases = tf.get_variable('b', [output_dim], 
            initializer=tf.constant_initializer(0.0), trainable=True)
        conv = tf.nn.bias_add(conv, biases)

        return conv

def snlinear(name, input_, output_size, seed=None, 
            use_bias=True, update_collection=None):
    shape = input_.shape.as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('W', [shape[1], output_size], tf.float32,
            initializer=tf.glorot_uniform_initializer(seed), trainable=True)        
        w_bar = spectral_normed_weight('sp_norm', w, update_collection=update_collection)
        mul = tf.matmul(input_, w_bar)
        if use_bias:
            bias = tf.get_variable('b', [output_size], 
                initializer=tf.constant_initializer(0.0), trainable=True)
            mul = tf.nn.bias_add(mul, bias)

        return mul