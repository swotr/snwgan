import os, sys
import time
import argparse
from tqdm import tqdm

import numpy as np
import cv2
import matplotlib
matplotlib.use('agg') # use matplotlib without having a window appear (you want this working in shell)
import matplotlib.pyplot as plt
import tensorflow as tf

from snops import *
from resnet import *
from IS.inception_score import *

IMAGE_HEIGHT, IMAGE_WIDTH = 32, 32 # original image size
CANON_HEIGHT, CANON_WIDTH = IMAGE_HEIGHT//1, IMAGE_WIDTH//1 # size to generate
BASE_HEIGHT, BASE_WIDTH = IMAGE_HEIGHT//8, IMAGE_WIDTH//8 # base size of generator
NUM_CLASSES = 10 # CIFAR10
AUX_DIM = 100
BASE_DIM = 32 # base number for channels (channels are multiple of this)
BATCH_SIZE = 64 # batch size for training and testing
Z_DIM = 128
N_CRITICS = 1 # number of D updates per each G update
EPOCHS = 130*N_CRITICS # roughly 100K G iterations
G_NAME = 'generator' # generator name for scoping
D_NAME = 'discriminator' # discriminator name for scoping

def one_hot(x):
    return np.squeeze(np.eye(NUM_CLASSES)[x.reshape(-1)])

class CifarData(object):
    '''
    CIFAR10 (32x32, 10 classes, 50K training, 60K total)
    0 = airplane, 1 = automobile, 2 = bird, 3 = cat, 4 = deer
    5 = dog, 6 = frog, 7 = horse, 8 = ship, 9 = truck
    '''
    def __init__(self, data_dir):
        def load(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                data = dict[b'data']
                labels = np.asarray(dict[b'labels'])
                images = data.reshape([-1, 3, IMAGE_HEIGHT*IMAGE_WIDTH])
                images = np.transpose(images, axes=[0, 2, 1])
                images = images.reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                labels = labels.reshape([-1, 1])
                return images, labels

        # Read 5 training batches
        images, labels = load(os.path.join(data_dir, 'data_batch_1'))
        self.images = images # np array
        self.labels = one_hot(labels) # np array
        for i in range(2, 6):
            images, labels = load(os.path.join(data_dir, 'data_batch_{}'.format(i)))
            self.images = np.vstack((self.images, images))
            self.labels = np.vstack((self.labels, one_hot(labels)))
        # Read test batch
        #images, labels = load(os.path.join(data_dir, 'test_batch'))
        #self.images = np.vstack((self.images, images))
        #self.labels = np.vstack((self.labels, labels))
        print('images (numpy): {}'.format(self.images.shape))
        print('labels (numpy): {}'.format(self.labels.shape))

        # Read inception logits
        #self.feats = np.load('/home/minje/dev/dataset/cifar/cifar10_inception_logit.npy')
        self.feats = np.load('/home/minje/dev/dataset/cifar/cifar10_inception_logit_reduced.npy')
        print('feats (numpy): {}'.format(self.feats.shape))

    def size(self):
        return len(self.labels)

    def get_data(self):
        #return self.images, self.labels
        return self.images, self.feats

    def get_data_with_label_noise(self, ratio):
        labels = np.argmax(self.labels, axis=1)
        n = len(labels)
        m = int(ratio*n)
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        idxs = idxs[:m] # index for adding label noise
        offs = np.random.randint(NUM_CLASSES, size=m) # label noise
        labels[idxs] = (labels[idxs]+offs)%NUM_CLASSES
        for idx in idxs: # random can't do all
            if labels[idx] == np.argmax(self.labels[idx]):
                labels[idx] = (labels[idx]+1)%NUM_CLASSES
        return self.images, one_hot(labels)

class CifarGanModel(object):
    def __init__(self):
        pass

    def generator(self, z, is_training):
        # Standard network
        # l = z
        # l = dense(l, BASE_HEIGHT*BASE_WIDTH*BASE_DIM*16, name='fc_in')
        # l = tf.reshape(l, [-1, BASE_HEIGHT, BASE_WIDTH, BASE_DIM*16])        
        # l = tf.nn.relu(batch_norm(deconv(l, BASE_DIM*8, 4, 2, name='deconv1'), is_training))        
        # l = tf.nn.relu(batch_norm(deconv(l, BASE_DIM*4, 4, 2, name='deconv2'), is_training))        
        # l = tf.nn.relu(batch_norm(deconv(l, BASE_DIM*2, 4, 2, name='deconv3'), is_training))        
        # l = conv(l, 3, 3, 1, name='conv_out')
        # img = tf.nn.tanh(l, 'gen_out')
    
        # ResNet (from 'Improved Training of WGAN ...')
        l = z
        l = dense(l, BASE_HEIGHT*BASE_WIDTH*BASE_DIM*8, name='fc_in')
        l = tf.reshape(l, [-1, BASE_HEIGHT, BASE_WIDTH, BASE_DIM*8])
        l = resnet_block('res1', l, BASE_DIM*8, 3, 'up', is_training)
        l = resnet_block('res2', l, BASE_DIM*8, 3, 'up', is_training)
        l = resnet_block('res3', l, BASE_DIM*8, 3, 'up', is_training)                
        l = tf.nn.relu(batch_norm(l, is_training))
        r = l
        l = conv(l, 3, 3, 1, name='conv_out')
        img = tf.nn.tanh(l, name='gen_out')

        # r = conv(r, BASE_DIM*8, 7, 4, name='r1')
        # r = tf.nn.relu(batch_norm(r, is_training))
        # r = tf.reshape(r, [-1, np.prod(r.shape.as_list()[1:])])
        # r = tf.nn.relu(batch_norm(dense(r, BASE_DIM*8, name='r2'), is_training))        
        # r = tf.layers.dropout(r, 0.5, training=is_training)
        # r = tf.nn.relu(batch_norm(dense(r, BASE_DIM*8, name='r3'), is_training))        
        # r = tf.layers.dropout(r, 0.5, training=is_training)
        # #r = dense(r, NUM_CLASSES, name='r4')
        # r = dense(r, AUX_DIM, name='r4')
        # lab = tf.nn.softmax(r, name='lab_out')

        r = resnet_block('r1', r, BASE_DIM*8, 3, 'down', is_training)
        r = resnet_block('r2', r, BASE_DIM*8, 3, 'down', is_training)
        r = resnet_block('r3', r, BASE_DIM*8, 3, 'same', is_training)
        r = resnet_block('r4', r, BASE_DIM*8, 3, 'same', is_training)
        r = tf.nn.relu(batch_norm(r, is_training))        
        r = tf.reduce_sum(r, axis=[1, 2])
        r = tf.nn.relu(batch_norm(dense(r, BASE_DIM*8, name='r5'), is_training))
        r = tf.layers.dropout(r, 0.5, training=is_training)
        r = dense(r, AUX_DIM, name='r6')
        lab = tf.nn.softmax(r, name='lab_out')
        
        return img, lab
    
    def discriminator(self, images, labels, update_collection):    
        l = images
        
        # Standard network
        # norm_op = tf.identity
        # l = tf.nn.leaky_relu(norm_op(snconv2d('sn_conv1', l, BASE_DIM*2, 3, 1, seed=np.sqrt(2), update_collection=update_collection)), 0.1)
        # l = tf.nn.leaky_relu(norm_op(snconv2d('sn_conv2', l, BASE_DIM*2, 4, 2, seed=np.sqrt(2), update_collection=update_collection)), 0.1)
        # l = tf.nn.leaky_relu(norm_op(snconv2d('sn_conv3', l, BASE_DIM*4, 3, 1, seed=np.sqrt(2), update_collection=update_collection)), 0.1)
        # l = tf.nn.leaky_relu(norm_op(snconv2d('sn_conv4', l, BASE_DIM*4, 4, 2, seed=np.sqrt(2), update_collection=update_collection)), 0.1)
        # l = tf.nn.leaky_relu(norm_op(snconv2d('sn_conv5', l, BASE_DIM*8, 3, 1, seed=np.sqrt(2), update_collection=update_collection)), 0.1)
        # l = tf.nn.leaky_relu(norm_op(snconv2d('sn_conv6', l, BASE_DIM*8, 4, 2, seed=np.sqrt(2), update_collection=update_collection)), 0.1)
        # l = tf.nn.leaky_relu(norm_op(snconv2d('sn_conv7', l, BASE_DIM*16, 3, 1, seed=np.sqrt(2), update_collection=update_collection)), 0.1)
        # l = tf.reshape(l, [-1, np.prod(l.shape.as_list()[1:])])
        # l = snlinear('logit', l, 1, seed=None, update_collection=update_collection)
        # return l
        
        # ResNet (from 'Improved Training of WGAN ...')
        l = snresnet_block_D1('res1', l, BASE_DIM*4, 3, 'down', update_collection=update_collection)
        l = snresnet_block('res2', l, BASE_DIM*4, 3, 'down', update_collection=update_collection)
        l = snresnet_block('res3', l, BASE_DIM*4, 3, 'same', update_collection=update_collection)
        l = snresnet_block('res4', l, BASE_DIM*4, 3, 'same', update_collection=update_collection)
        l = tf.nn.relu(l)
        l = tf.reduce_sum(l, axis=[1, 2]) # global sum pooling
        x = snlinear('logit', l, 1, seed=None, update_collection=update_collection)
        # label embedding
        e = snembed('label_embed', labels, BASE_DIM*4, update_collection=update_collection) 
        y = tf.reduce_sum(e*l, axis=1) # inner product of word vector and feature
        return x+y
        
    def input(self):
        # Create placeholder for input
        self.images = tf.placeholder(tf.uint8, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='images')
        #self.labels = tf.placeholder(tf.float32, (None, NUM_CLASSES), name='labels') # assume one-hot form        
        self.labels = tf.placeholder(tf.float32, (None, AUX_DIM), name='labels') # assume one-hot form        
        self.z = tf.placeholder(tf.float32, (None, Z_DIM), name='z')
        return self.images, self.labels, self.z

    def build_model(self, is_training=True):        
        # Input placeholders
        images, labels, z = self.input()        
        
        # Convert image from uint8 to float32.
        # Normalize.
        images = tf.cast(images, dtype=tf.float32)
        images = images/127.5-1

        with tf.variable_scope(G_NAME, reuse=tf.AUTO_REUSE):
            fakes, fakes_lab = self.generator(z, is_training)
            tf.summary.image('real', images, 3)
        tf.summary.image('fake', fakes, 30)
        
        with tf.variable_scope(D_NAME, reuse=tf.AUTO_REUSE): # what happens in BN reuse?
            logits_real = self.discriminator(images, labels, update_collection=None)
            logits_fake = self.discriminator(fakes, fakes_lab, update_collection='NO_OPS')

        # Standard WGAN loss
        #self.d_loss = tf.reduce_mean(tf.nn.softplus(logits_fake)+tf.nn.softplus(-logits_real))        
        #self.g_loss = tf.reduce_mean(tf.nn.softplus(-logits_fake))
        # Hinge WGAN loss
        self.d_loss = tf.reduce_mean(tf.nn.relu(1.0-logits_real)+tf.nn.relu(1.0+logits_fake))
        self.g_loss = -tf.reduce_mean(logits_fake)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)        
        
        # Collect variables        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, D_NAME)
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, G_NAME)

        # Collect summaries
        self.summary = tf.summary.merge_all()

    def get_optimizers(self):
        ''' This is for single GPU.
            You don't need to aggregate gradients. 
            So we use minimize() '''
        d_opt = tf.train.AdamOptimizer(0.0004, beta1=0.5, beta2=0.999, epsilon=1e-8)
        g_opt = tf.train.AdamOptimizer(0.0001, beta1=0.5, beta2=0.999, epsilon=1e-8)
        d_min = d_opt.minimize(self.d_loss, var_list=self.d_vars)
        g_min = g_opt.minimize(self.g_loss, var_list=self.g_vars)        
        return d_min, g_min

class CifarGanTrainer(object):
    def __init__(self):
        pass

    def train(self, data, model, log_dir):
        # Build GAN model & optimizer
        model.build_model()
        d_min, g_min = model.get_optimizers()

        # Prepare dataset
        # Create placeholders for efficient batch feed with tf.dataset
        train_images, train_labels = data.get_data() # load numpy array        
        #train_images, train_labels = data.get_data_with_label_noise(0.0) # load numpy array
        dataset = tf.data.Dataset.from_tensor_slices((model.images, model.labels))
        dataset = dataset.prefetch(BATCH_SIZE*10)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset_iter = dataset.make_initializable_iterator()
        next_batch = dataset_iter.get_next()

        # Make a summary writer
        train_writer = tf.summary.FileWriter(log_dir)
        iter_counter = 0

        # Make a train saver (for checkpoint)
        train_saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

        with tf.Session() as sess:
            # initialize all variables to be trained
            sess.run(tf.global_variables_initializer())

            # training body
            for epoch in range(EPOCHS):
                # (re)initialize dataset iterator
                sess.run(dataset_iter.initializer, feed_dict={model.images: train_images,  
                                                              model.labels: train_labels})
                d_loss, g_loss = 0, 0
                n_iter = train_images.shape[0]//(BATCH_SIZE*N_CRITICS)
                with tqdm(total=n_iter) as pbar:
                    while True:
                        try:
                            # optimize D --> G
                            # make merged summary when training G
                            for _ in range(N_CRITICS):
                                # make a batch
                                image_batch, label_batch = sess.run(next_batch)
                                feed_dict = {model.images: image_batch,
                                             model.labels: label_batch,
                                             model.z: np.random.randn(BATCH_SIZE, Z_DIM).astype(np.float32)}
                                _, _d_loss = sess.run([d_min, model.d_loss], feed_dict=feed_dict)
                            # Reuse the last batch (since G doesn't care)
                            # Make sure that z is newly sampled
                            feed_dict = {model.images: image_batch,
                                         model.labels: label_batch,
                                         model.z: np.random.randn(BATCH_SIZE, Z_DIM).astype(np.float32)}
                            _, _g_loss, summary = sess.run([g_min, model.g_loss, model.summary], feed_dict=feed_dict)
                            d_loss += _d_loss
                            g_loss += _g_loss
                            iter_counter += 1
                            pbar.update(1)
                        except tf.errors.OutOfRangeError:
                            break
                d_loss /= n_iter
                g_loss /= n_iter

                # Write summary & save the model.
                train_writer.add_summary(summary, iter_counter)
                train_saver.save(sess, os.path.join(log_dir, 'model'), global_step=iter_counter)
                print('epoch={:03d}/{:03d}, d_loss={:.4f}, g_loss={:.4f}'.format(
                    (epoch+1), EPOCHS, d_loss, g_loss))

                # # Save a snapshot of generated images per each epoch.                
                # img_tensor = tf.get_default_graph().get_tensor_by_name(os.path.join(G_NAME, 'gen_out:0'))
                # lab_tensor = tf.get_default_graph().get_tensor_by_name(os.path.join(G_NAME, 'lab_out:0'))
                # imgs, labs = sess.run([img_tensor, lab_tensor], 
                #     feed_dict={model.z: np.random.randn(100, Z_DIM).astype(np.float32)})
                # imgs = 127.5*(imgs+1)
                # imgs = np.clip(imgs, 0, 255).astype(np.uint8)                
                # # one-hot to scalar
                # labs = np.argmax(labs, axis=1).astype(np.int32)
                # # Make and save snapshot.
                # fig, axs = plt.subplots(NUM_CLASSES, 30, figsize=(30, NUM_CLASSES))
                # plt.subplots_adjust(hspace=0, wspace=0)
                # his = np.zeros(NUM_CLASSES).astype(np.int32)
                # for ax in axs.flat:
                #     ax.axis('off')
                # for i in range(imgs.shape[0]):
                #     row = labs[i]
                #     col = his[row]
                #     axs[row][col].imshow(imgs[i])
                #     his[row] = min(his[row]+1, 29)
                # fpath = os.path.join(log_dir, '{0:03d}.png'.format(epoch+1))
                # plt.savefig(fpath)
                # plt.close()
                
                # Calculate Inception Score at every XX epoch
                # if epoch%10 == 0:
                #     repeats = (50000+BATCH_SIZE-1)//BATCH_SIZE
                #     for _ in tqdm(range(repeats)):
                #         feed_dict = {model.labels: np.zeros([BATCH_SIZE, 1], np.uint8),
                #                     model.z: np.random.randn(BATCH_SIZE, Z_DIM).astype(np.float32)}
                #         target_tensor = graph.get_tensor_by_name(os.path.join(G_NAME, 'gen_out:0'))
                #         images = ses.run(target_tensor, feed_dict=feed_dict)
                #         images = 127.5*(images+1)
                #         images = np.clip(images, 0, 255).astype(np.uint8)
                
class CifarGanTester(object):
    def __init__(self):
        pass

    def test(self, model_path, test_dir):
        # Load checkpoint model
        # - We're not using '.meta' for importing graph to control 'is_training' parameter
        # - We use build_model() to build a model from scratch
        # - Then restore model parameters  
        tf.reset_default_graph()
        model = CifarGanModel()
        model.build_model(is_training=False)

        saver = tf.train.Saver()
        with tf.Session() as sess:            
            saver.restore(sess, model_path)
            graph = tf.get_default_graph()

            uid = 0 # unique id of each generated image
            repeats = (50000+BATCH_SIZE-1)//BATCH_SIZE
            for _ in tqdm(range(repeats)):
                feed_dict = {model.z: np.random.randn(BATCH_SIZE, Z_DIM).astype(np.float32)}
                img_tensor = graph.get_tensor_by_name(os.path.join(G_NAME, 'gen_out:0'))
                lab_tensor = graph.get_tensor_by_name(os.path.join(G_NAME, 'lab_out:0'))
                imgs, labs = sess.run([img_tensor, lab_tensor], feed_dict=feed_dict) # get generated images and labels
                imgs = 127.5*(imgs+1)
                imgs = np.clip(imgs, 0, 255).astype(np.uint8)                
                labs = np.argmax(labs, axis=1).astype(np.int32) # one-hot to scalar
                for i in range(imgs.shape[0]):
                    fpath = os.path.join(test_dir, '{0:05d}_{1:03d}.jpg'.format(uid, labs[i]))
                    cv2.imwrite(fpath, cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
                    uid += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--test', help='generate images')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.test:
        test_dir = '/home/minje/dev/dataset/cifar/cifar-10-fake/'
        tester = CifarGanTester()
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir) # remove all existing files
        os.mkdir(test_dir)
        tester.test(args.load, test_dir)
    else:
        data = CifarData('/home/minje/dev/dataset/cifar/cifar-10-batches-py/')
        model = CifarGanModel()
        trainer = CifarGanTrainer()
        fname = os.path.basename(sys.argv[0]) # argv[0] contains python filename
        fname = fname[:fname.rfind('.')]
        log_dir = os.path.join('train_log/', fname)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        trainer.train(data, model, log_dir)
