"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
Modified for dialog processing
"""
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataset import Dataset
import torch

class DialogDataset(Dataset):
    def __init__(self, data):
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.token_type_ids = data['token_type_ids']
        self.act = data['act']
        self.emotion = data['emotion']
        self.dialog = data['dialog']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'act': self.act[idx],
            'emotion': self.emotion[idx],
            'dialog': self.dialog[idx]
        }

class DataLoader:
    def __init__(self):
        print('dataset initializing start')
        
    def make_dataset(self, train_data, valid_data, test_data):
        """
        Create Dataset objects from the data
        """
        self.train_dataset = DialogDataset(train_data)
        self.valid_dataset = DialogDataset(valid_data)
        self.test_dataset = DialogDataset(test_data)
        
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Valid size: {len(self.valid_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")
        
        return self.train_dataset, self.valid_dataset, self.test_dataset

    def build_vocab(self, train_data, min_freq):
        """
        No need to build vocabulary as we're using pre-tokenized data
        Keeping method for compatibility
        """
        pass

    def make_iter(self, train, validate, test, batch_size, device):
        """
        Create data iterators
        """
        train_iterator = TorchDataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if device == "cuda" else False
        )
        
        valid_iterator = TorchDataLoader(
            validate,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if device == "cuda" else False
        )
        
        test_iterator = TorchDataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if device == "cuda" else False
        )
        
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
