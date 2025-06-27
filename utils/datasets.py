import torch
from torch.utils.data import random_split

def split_dataset(dataset, 
                  train_ratio=0.9, valid_ratio=0.1, test_ratio=None):
    # Split into 2 cases, with test data and without test data
    length = dataset.__len__()
    if test_ratio == None:
        train_len = int(length * train_ratio)
        valid_len = length - train_len
        train_dataset, valid_dataset = random_split(dataset,
                                                    [train_len, valid_len],
                                                    generator=torch.Generator().manual_seed(731))
        return train_dataset, valid_dataset, None
    
    else:
        train_len = int(length * train_ratio)
        valid_len = int(length * valid_ratio)
        test_len = length - train_len - valid_len
        train_dataset, valid_dataset, test_dataset = random_split(dataset,
                                                                  [train_len, valid_len, test_len],
                                                                  generator=torch.Generator().manual_seed(731))
        return train_dataset, valid_dataset, test_dataset

