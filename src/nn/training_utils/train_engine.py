import torch
import torch.nn as nn
from tqdm import tqdm
import os
from .dice_loss import DiceLoss


def to_device(batch: dict, device: str):
    return {k: v.to(device) for k, v in batch.items()}


class Trainer:
    def __init__(self, model,
                 optimizer,
                 train_dataloader,
                 val_dataloader,
                 device: str,
                 scheduler=None,
                 use_weighted=False):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        if use_weighted:
            class_weights = torch.ones((9,), device=device)
            class_weights[0] = 0.5
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, epochs: int, save_path, val_every: int = 1, accumulation_step: int = 1):
        assert epochs % val_every == 0, 'Epochs number should be divisible by \'val_every\' parameter'
        assert accumulation_step > 0, '\'accumulation_step\' parameter should be greater than zero'
        os.makedirs(save_path, exist_ok=True)
        losses = []
        print('Start training')
        self.model.train()
        for epoch in range(epochs):
            mean_loss = 0
            for it, inputs in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc='Training'):
                inputs = to_device(inputs, self.device)
                labels = inputs.pop('labels')

                logits = self.model(**inputs).logits
                num_targets = logits.shape[-1]
                loss = self.criterion(logits.view(-1, num_targets), labels.flatten()) / accumulation_step

                loss.backward()

                mean_loss += loss.item() / len(self.train_dataloader)
                if (it + 1) % accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            losses.append(mean_loss)
            print(f'Epoch: {epoch}, Mean loss: {mean_loss}')
            torch.save(self.model, os.path.join(save_path, 'best_model.pth'))

        return losses
