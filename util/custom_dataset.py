import torch
from torch.utils.data import Dataset

class DialogDataset(Dataset):
    def __init__(self, input_ids, attention_masks, acts, emotions):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.acts = acts
        self.emotions = emotions

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'act': self.acts[idx],
            'emotion': self.emotions[idx]
        } 