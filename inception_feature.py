import os, sys
import functools
import time
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan


class InceptionScore():
    def __init__(self, batch_size):
        self.batch_size = batch_size         
        self.inception_images = tf.placeholder(tf.float32, [self.batch_size, 3, None, None])
        self.logits = self.inception_logits()

    def inception_logits(self):
        images = self.inception_images
        images = tf.transpose(images, [0, 2, 3, 1])
        size = 299
        images = tf.image.resize_bilinear(images, [size, size])
        generated_images_list = array_ops.split(images, num_or_size_splits=1)
        logits = functional_ops.map_fn(
            fn = functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
            elems = array_ops.stack(generated_images_list),
            parallel_iterations = 1,
            back_prop = False,
            swap_memory = True,
            name = 'RunClassifier')
        logits = array_ops.concat(array_ops.unstack(logits), 0)
        return logits

    def get_inception_probs(self, sess, inps):
        n_batches = len(inps) // self.batch_size
        preds = np.zeros([n_batches * self.batch_size, 1000], dtype=np.float32)
        for i in tqdm(range(n_batches)):
            inp = inps[i * self.batch_size : (i + 1) * self.batch_size] / 127.5 - 1
            preds[i * self.batch_size : (i + 1) * self.batch_size] = sess.run(self.logits, feed_dict={self.inception_images: inp})[:, :1000]
        preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
        return preds

    def preds2score(self, preds):
        scores = []
        splits = 1
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

    def get_inception_score(self, images):
        assert(type(images) == np.ndarray)
        assert(len(images.shape) == 4)
        assert(images.shape[1] == 3)
        assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
        print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], 1))
        start_time = time.time()
        with tf.Session() as sess:
            preds = self.get_inception_probs(sess, images)
        mean, std = self.preds2score(preds)
        print('Inception Score calculation time: %f s' % (time.time() - start_time))
        return mean, std # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits.

def get_images_from_files(path):
    import cv2
    images = np.empty(shape=[50000, 3, 32, 32], dtype=np.uint8)
    idx = 0
    for root, dir, files in os.walk(path):
        for file in files:
            if file.endswith(tuple(['.jpg', '.png', 'bmp'])):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                img = img[:, :, (2, 1, 0)] # BGR to RGB                
                img = np.transpose(img, (2, 0, 1)) # RGB, H, W                    
                images[idx] = img
                idx += 1
                if idx >= 50000:
                    break                
    print('images.shape: {}'.format(images.shape))
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated GPU list to use')
    parser.add_argument('--data_dir', help='path to data folder')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    

    images = get_images_from_files(args.data_dir)        
    mean, std = InceptionScore(64).get_inception_score(images)
    print('IS: ', mean, std)
