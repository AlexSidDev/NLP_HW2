import pandas as pd
import torch
import torch.nn.functional as F


class NERDataset:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        tokens = torch.tensor(eval(self.data['tokens'][index]), dtype=torch.long)
        labels = torch.tensor(eval(self.data['labels'][index]), dtype=torch.long)
        return tokens, labels


class DataCollator:
    def __init__(self, max_len: int, token_pad_id: int, label_pad_id: int = -1):
        self.max_len = max_len
        self.pad_id = token_pad_id
        self.label_pad_id = label_pad_id
        self.pad_fn = lambda sample, value: F.pad(sample, pad=(0, self.max_len - len(sample)), value=value)

    def __call__(self, data):
        batch = dict()
        padded_masks = [self.pad_fn(torch.ones_like(sample[0]), 0) for sample in data]
        batch['attention_mask'] = torch.stack(padded_masks)

        padded_ids = [self.pad_fn(sample[0], self.pad_id) for sample in data]
        batch['input_ids'] = torch.stack(padded_ids)

        padded_labels = [self.pad_fn(sample[1], self.label_pad_id) for sample in data]
        batch['labels'] = torch.stack(padded_labels)
        return batch

