import seqeval
import torch
import numpy as np
import evaluate
from tqdm import tqdm


def to_device(batch: dict, device: str):
    return {k: v.to(device) for k, v in batch.items()}


def preds_to_bio(preds: list, word_inds: list, labels_mapping: dict):
    bio_preds = []
    previous_ind = None
    for i, pred in enumerate(preds):
        if word_inds[i] is None or word_inds[i] == previous_ind:
            continue
        bio_preds.append(labels_mapping[pred])
        previous_ind = word_inds[i]
    return bio_preds


def inference(model, val_dataloader, word_inds, labels_mapping, device='cuda'):
    preds = []
    for it, inputs in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        with torch.no_grad():
            inputs = to_device(inputs, device)
            inputs.pop('labels')

            outputs = model(**inputs).logits.argmax(dim=-1).cpu().numpy().tolist()
            preds.extend(outputs)

    bio_preds = []
    for i, row in enumerate(preds):
        bio_preds.append(preds_to_bio(row, word_inds[i], labels_mapping))

    return bio_preds


