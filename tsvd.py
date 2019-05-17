import numpy as np
from sklearn.decomposition import TruncatedSVD

trg_dim = 128
org_file = '/home/minje/dev/dataset/stl/stl_unlabeled_inception_pool_3.npy'
trg_file = '/home/minje/dev/dataset/stl/stl_unlabeled_inception_pool_3_reduced128.npy'
# trg_dim = 32
# org_file = '/home/minje/dev/dataset/cifar/cifar_unlabeled_inception_pool_3.npy'
# trg_file = '/home/minje/dev/dataset/cifar/cifar_unlabeled_inception_pool_3_reduced32.npy'

data = np.load()

svd = TruncatedSVD(n_components=128)
svd.fit(data)
print(svd.explained_variance_ratio_.sum())
data_reduced = svd.transform(data)

np.save(trg_file)

# for i in range(10):
#     feat = data_reduced[i]
#     print(np.sqrt(np.sum(feat**2)))

