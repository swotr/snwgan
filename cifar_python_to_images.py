import os
import numpy as np
import cv2

cifar10_path = '/home/minje/dev/dataset/cifar/cifar-10-batches-py'
cifar10_images = '/home/minje/dev/dataset/cifar/cifar-10-images'

def load_cifar10(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        data = dict[b'data']
        labels = dict[b'labels']
        images = data.reshape([-1, 3, 32*32])
        images = np.transpose(images, axes=[0, 2, 1])
        images = images.reshape([-1, 32, 32, 3])
        images = images.astype(np.float32)
        return images, labels

def write_cifar10_jpg(images, labels, cnt):
    '''
    images : numpy array of [n_images, height, width, 3]
    labels : list of labels (0-9)
    cnt : global counter for unique name
    '''
    n_images = images.shape[0]    
    for i in range(n_images):
        fpath = os.path.join(cifar10_images, 'img_{0:05d}_{1:03d}.jpg'.format(cnt[0], labels[i]))
        cv2.imwrite(fpath, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
        cnt[0] += 1

# cifar10
# names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
# cnt = [1]
# for name in names:
#     images, labels = load_cifar10(os.path.join(cifar10_path, name))
#     print(name, images.shape)
#     write_cifar10_jpg(images, labels, cnt)

cifar100_path = '/home/minje/dev/dataset/cifar/cifar100/cifar-100-python'
cifar100_images = '/home/minje/dev/dataset/cifar/cifar100/cifar-100-images'

def load_cifar100(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        data = dict[b'data']
        coarse_labels = dict[b'coarse_labels']
        fine_labels = dict[b'fine_labels']
        images = data.reshape([-1, 3, 32*32])
        images = np.transpose(images, axes=[0, 2, 1])
        images = images.reshape([-1, 32, 32, 3])
        images = images.astype(np.float32)
        return images, coarse_labels, fine_labels

def write_cifar100_jpg(images, coarse_labels, fine_labels, cnt):
    '''
    images : numpy array of [n_images, height, width, 3]
    coarse_labels : list of labels [0,20)
    fine_labels : list of labels [0, 100)
    cnt : global counter for unique name
    '''
    n_images = images.shape[0]    
    for i in range(n_images):
        fpath = os.path.join(cifar100_images, 
                'img_{0:05d}_{1:03d}_{2:03d}.jpg'.format(cnt[0], coarse_labels[i], fine_labels[i]))
        cv2.imwrite(fpath, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
        cnt[0] += 1


names = ['train']
cnt = [1]
for name in names:
        images, coarse_labels, fine_labels = load_cifar100(os.path.join(cifar100_path, name))
        print(name, images.shape)
        write_cifar100_jpg(images, coarse_labels, fine_labels, cnt)
