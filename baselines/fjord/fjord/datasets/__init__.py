from .fl_cifar10 import FLCifar10
from .fl_cifar100 import FLCifar100
from .wrappers import Cifar10_lda, Cifar100_lda
from .femnist import FEMNIST
from .shakespeare import Shakespeare, SHAKESPEARE_EVAL_BATCH_SIZE

FEDERATED_DATASETS = ['shakespeare', 'femnist', 'cifar100_fl', 'cifar10_lda', 'cifar100_lda']


def get_num_clients(dataset):
    dataset = dataset.lower()
    if dataset == 'shakespeare':
        num_clients = 715
    elif dataset == 'femnist':
        num_clients = 3400
    elif dataset == 'cifar100_fl':
        num_clients = 500
    elif dataset == 'cifar10_fl':
        num_clients = 100
    elif dataset == 'cifar10_lda':
        num_clients = 100
    elif dataset == 'cifar100_lda':
        num_clients = 500
    else:
        raise ValueError(f"Dataset {dataset} is either not supported or not federated.")
    return num_clients

