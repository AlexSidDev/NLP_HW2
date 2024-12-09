import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import PreTrainedTokenizer

from .ner_dataset import NERDataset


class DatasetTokenizer:
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int, labels_mapping: dict):
        self.tokenizer = tokenizer
        self.labels_mapping = labels_mapping
        self.data = data
        self.max_len = max_len

    def re_tokenize_row(self, tokens, labels):
        row_tokens = []
        row_labels = []
        for tok_ind, token in enumerate(tokens):
            label = labels[tok_ind]
            if label.startswith('B-') or label.startswith('I-'):
                label = label.split('-')[-1]
            label = self.labels_mapping[label]
            ids_list = self.tokenizer.encode(token)
            row_tokens.extend(ids_list)
            row_labels.extend([label] * len(ids_list))
        if len(row_tokens) > self.max_len:
            slice_index = 0
            rows = []
            while slice_index < len(row_tokens):
                sls = slice(slice_index, min(slice_index + self.max_len, len(row_tokens)))
                rows.append([row_tokens[sls], row_labels[sls]])
                slice_index += self.max_len
            return rows
        else:
            return [[row_tokens, row_labels]]

    def re_tokenize(self):
        tokens = self.data['tokens']
        labels = self.data['labels']
        processed_rows = []
        for row in range(len(tokens)):
            row_tokens = tokens[row]
            row_labels = labels[row]
            processed_row = self.re_tokenize_row(row_tokens, row_labels)
            processed_rows.extend(processed_row)
        return pd.DataFrame(processed_rows, columns=['tokens', 'labels'], dtype='object')


def create_dataset(path, save_file_name, tokenizer: PreTrainedTokenizer, max_len: int,
                   labels_mapping: dict, force_recreate=False):
    if not os.path.exists(save_file_name) or force_recreate:
        print("Start data processing...")
        processed_data = None
        raw_data = pd.read_csv(path)
        re_tokenizer = DatasetTokenizer(raw_data, tokenizer, max_len, labels_mapping)
        if processed_data is not None:
            processed_data = re_tokenizer.re_tokenize()
        else:
            processed_data = pd.concat([processed_data, re_tokenizer.re_tokenize()], ignore_index=True)
        processed_data.to_csv(save_file_name)
    else:
        print("Found cached data in", save_file_name)
    all_data = pd.read_csv(save_file_name)
    return all_data



