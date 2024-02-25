import os
import h5py

from .fl_dataset import FLDataset
from torchvision.datasets.utils import download_url

TFF_DATASETS = {
    'cifar100_fl': 'https://storage.googleapis.com/tff-datasets-public/fed_cifar100.tar.bz2',
    'femnist': 'https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2',
    'shakespeare': 'https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2'
}


class H5TFFDataset(FLDataset):
    def __init__(self, h5_path, client_id, dataset_name, data_key):
        self.h5_path = h5_path
        if not os.path.isfile(h5_path):
            one_up = os.path.dirname(h5_path)
            target = os.path.basename(TFF_DATASETS[dataset_name])
            download_url(TFF_DATASETS[dataset_name], one_up)

            def extract_bz2(filename, path="."):
                import tarfile
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(path)
            extract_bz2(os.path.join(one_up, target), one_up)

            # raise ValueError(f'File not found at {h5_path},
            # dataset available at: {TFF_DATASETS[dataset_name]}')
        self.dataset = None
        self.clients = list()
        self.clients_num_data = dict()
        self.client_and_indices = list()
        with h5py.File(self.h5_path, 'r') as file:
            data = file['examples']
            for client in list(data.keys()):
                self.clients.append(client)
                n_data = len(data[client][data_key])
                for i in range(n_data):
                    self.client_and_indices.append((client, i))
                self.clients_num_data[client] = n_data
        self.num_clients = len(self.clients)
        self.length = len(self.client_and_indices)

        self.set_client(client_id)

    def set_client(self, index=None):
        if index is None:
            self.client_id = None
            self.length = len(self.client_and_indices)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = self.clients_num_data[self.clients[index]]

    def _get_item_preprocess(self, index):
        # loading in getitem allows us to use multiple processes for data loading
        # because hdf5 files aren't pickelable so can't transfer them across processes
        # https://discuss.pytorch.org/t/hdf5-a-data-format-for-pytorch/40379
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')["examples"]
        if self.client_id is None:
            client, i = self.client_and_indices[index]
        else:
            client, i = self.clients[self.client_id], index
        return client, i

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.length
