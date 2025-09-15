import pathlib
import torch

from torch.utils.data import random_split
from torchvision import datasets
import pytorch_lightning as pl
from hydra.utils import instantiate
from iterativenn.utils.DatasetUtils import AdditionImageSequence, GymImageSequence, ImageSequence, RandomImageSequence
from iterativenn.utils.DatasetUtils import get_transform
from iterativenn.utils.operations import trivial_collate_fn

from iterativenn.lit_modules.IteratedModel import IteratedModel, ConfigCallbacks

class WrapperDataModule(pl.LightningDataModule):
    def __init__(self, dl_train, dl_val, dl_test):
        super().__init__()
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.dl_test = dl_test

    def train_dataloader(self):
        return self.dl_train
    
    def val_dataloader(self):
        return self.dl_val

    def test_dataloader(self):
        return self.dl_test

def DataFactory(cfg):
    """
    Creates pl.LightningDataModules based on several configurations and returns 
    the object based on client's requirement
    """
    data_dir = pathlib.Path.home() / 'work' / 'data'
    dataset_str = cfg['dataset']
    transform = get_transform(cfg['transform'])

    """
    Uses dataset_str to create torch.dataset
    Returns: torch dataset
    """
    if dataset_str == 'MNIST':
        dataset = datasets.MNIST(data_dir, download=True, transform=transform)
    elif dataset_str == 'EMNIST':
        dataset = datasets.EMNIST(data_dir, split='digits', download=True, transform=transform)
    elif dataset_str == 'FashionMNIST':
        dataset = datasets.FashionMNIST(data_dir, download=True, transform=transform)
    elif dataset_str== 'KMNIST':
        dataset = datasets.KMNIST(data_dir, download=True, transform=transform)
    elif dataset_str == 'cartpole':
        dataset = GymImageSequence(n_data_points=2000,
                                    min_copies=4,
                                    max_copies=cfg['max_copies'],
                                    env_str=cfg['env_str'])
    elif dataset_str == 'lunar':
        dataset = GymImageSequence(n_data_points=2000,
                                    min_copies=4,
                                    max_copies=cfg['max_copies'],
                                    env_str=cfg['env_str'])
    elif dataset_str == 'acrobot':
        dataset = GymImageSequence(n_data_points=2000,
                                    min_copies=4,
                                    max_copies=cfg['max_copies'],
                                    env_str=cfg['env_str'])
    else:
        raise ValueError('Unknown dataset %s' % dataset_str)

 
    """
    transforms dataset to create another torch.dataset
    Returns: torch dataset (sequence)
    """
    is_sequence = False
    if cfg.get('sequence', False):
        dataset = instantiate(cfg.sequence_type, dataset)
        is_sequence = True

    if cfg.get('sequence_dict', False):
        is_sequence = True
        if cfg['sequence_dict']['type'] == 'uniform':
            dataset = ImageSequence(dataset,
                                    min_copies=cfg['sequence_dict']['min_copies'],
                                    max_copies=cfg['sequence_dict']['max_copies'],
                                    evaluate_loss=cfg['sequence_dict']['evaluate_loss']
                                   )
        elif cfg['sequence_dict']['type'] == 'random':
            dataset = RandomImageSequence(dataset,
                                          min_copies=cfg['sequence_dict']['min_copies'],
                                          max_copies=cfg['sequence_dict']['max_copies'],
                                          evaluate_loss=cfg['sequence_dict']['evaluate_loss']
                                         )
        elif cfg['sequence_dict']['type'] == 'addition':
            dataset = AdditionImageSequence(dataset, copies=cfg['sequence_dict']['copies'])
        elif cfg['sequence_dict']['type'] == 'pass':
            pass
        else:
            raise ValueError('Unknown sequence type %s' % cfg['sequence_type'])

    """
    Wraps torch.data.random_split
    Args:
        dataset:
    Returns:
        train, val, test, rest
    """
    train, val, test, rest = random_split(dataset,
                                        [cfg['train_size'],
                                        cfg['validation_size'],
                                        cfg['test_size'],
                                        len(dataset) - (cfg['train_size'] + cfg['validation_size'] +
                                                        cfg['test_size'])])

    if is_sequence:
        dl_train = torch.utils.data.DataLoader(train, num_workers=0,
                                            batch_size=cfg['batch_size'], 
                                            collate_fn = trivial_collate_fn)
        dl_val = torch.utils.data.DataLoader(val, num_workers=0,
                                            batch_size=cfg['batch_size'], 
                                            collate_fn = trivial_collate_fn)
        dl_test = torch.utils.data.DataLoader(test, num_workers=0,
                                            batch_size=cfg['batch_size'], 
                                            collate_fn = trivial_collate_fn)
    else:
        dl_train = torch.utils.data.DataLoader(train, num_workers=0,
                                            batch_size=cfg['batch_size'])
        dl_val = torch.utils.data.DataLoader(val, num_workers=0,
                                            batch_size=cfg['batch_size'])
        dl_test = torch.utils.data.DataLoader(test, num_workers=0,
                                            batch_size=cfg['batch_size'])
    return WrapperDataModule(dl_train, dl_val, dl_test)

