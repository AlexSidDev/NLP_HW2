from transformers import AutoModelForTokenClassification, AutoConfig
import torch.nn as nn


def create_model(model_name_or_path: str, labels_mapping: dict):
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, num_labels=len(labels_mapping),
                                                            id2label={v: k for k, v in labels_mapping.items()},
                                                            label2id=labels_mapping)
    return model
