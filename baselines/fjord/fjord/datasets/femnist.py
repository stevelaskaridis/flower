import os
import numpy as np
import torchvision

from .h5_tff_dataset import H5TFFDataset


class FEMNIST(H5TFFDataset):
    def __init__(self, h5_path, train=True, client_id=None):
        if train:
            h5_path = os.path.join(h5_path, 'femnist/fed_emnist_train.h5')
        else:
            h5_path = os.path.join(h5_path, 'femnist/fed_emnist_test.h5')
        super(FEMNIST, self).__init__(h5_path, client_id, 'femnist', 'pixels')
        self.transform = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        client, i = self._get_item_preprocess(index)
        x = 1 - self.transform(self.dataset[client]['pixels'][i])
        y = np.int64(self.dataset[client]['label'][i])
        return x, y
