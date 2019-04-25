import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

stl10_path = '/home/minje/dev/dataset/stl/stl10_binary' # source folder for extraction
stl10_image_folder = '/home/minje/dev/dataset/stl/images/unlabeled_48x48' # taret folder for extraction

def load_images(file):
    with open(file, 'rb') as fobj:
        images = np.fromfile(fobj, dtype=np.uint8)
        images = images.reshape([-1, 3, 96, 96])
        images = np.transpose(images, axes=[0, 3, 2, 1])
        return images

def load_labels(file):
    with open(file, 'rb') as fobj:
        labels = np.fromfile(fobj, dtype=np.uint8)
        return labels

def write_jpg(images, labels, cnt):
    '''
    images : numpy array of [n_images, height, width, 3]
    labels : numpy array of [n_images]; [0,10] ; STL-10 default is [1,10] ; 0 means unlabled
    cnt : global counter for unique name
    '''
    n_images = images.shape[0]
    for i in range(n_images):
        fpath = os.path.join(stl10_image_folder, 
            'img_{0:05d}_{1:03d}.jpg'.format(cnt[0], labels[i]))
        img = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (48, 48)) # resize from 96x96 to 48x48
        cv2.imwrite(fpath, img)
        cnt[0] += 1

# counter for unique name for images
cnt = [1]

# train data (500images/class)
# images = load_images(os.path.join(stl10_path, 'train_X.bin'))
# labels = load_labels(os.path.join(stl10_path, 'train_y.bin'))
# print(images.shape)
# write_jpg(images, labels, cnt)

# train data (800images/class)
# images = load_images(os.path.join(stl10_path, 'test_X.bin'))
# labels = load_labels(os.path.join(stl10_path, 'test_y.bin'))
# print(images.shape)
# write_jpg(images, labels, cnt)

# unlabled (100K)
images = load_images(os.path.join(stl10_path, 'unlabeled_X.bin'))
labels = np.zeros(images.shape[0]).astype(np.uint8)
print(images.shape)
write_jpg(images, labels, cnt)