from torch.utils.data import Dataset


class FLDataset(Dataset):

    def set_client(self, index=None):
        raise NotImplementedError
