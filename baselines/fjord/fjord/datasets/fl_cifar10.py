from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np


class FLCifar10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, client_id=None):

        super(FLCifar10, self).__init__(root, train=train, transform=transform,
                                        target_transform=target_transform, download=download)
        self.num_clients = 100
        self.dataset_indices = np.arange(len(self.data))
        np.random.seed(123)
        np.random.shuffle(self.dataset_indices)
        self.n_client_samples = len(self.data) // self.num_clients
        self.set_client(client_id)

    def set_client(self, index=None):
        if index is None:
            self.client_id = None
            self.length = len(self.data)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = self.n_client_samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.client_id is None:
            actual_index = index
        else:
            actual_index = int(self.client_id) * self.n_client_samples + index
        img, target = self.data[actual_index], self.targets[actual_index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length
