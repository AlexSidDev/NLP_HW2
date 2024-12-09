import torch
import torch.nn as nn
from tqdm import tqdm
import os


def to_device(batch: dict, device: str):
    return {k: v.to(device) for k, v in batch.items()}


class Trainer:
    def __init__(self, model, optimizer, train_dataloader, val_dataloader, metric, device: str, scheduler=None):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metric = metric

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, epochs: int, save_path, val_every: int = 1, accumulation_step: int = 1):
        assert epochs % val_every == 0, 'Epochs number should be divisible by \'val_every\' parameter'
        assert accumulation_step > 0, '\'accumulation_step\' parameter should be greater than zero'
        best_metric = 0
        all_metrics = []
        losses = []
        print('Start training')
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

            if (epoch + 1) % val_every == 0:
                self.model.eval()
                all_preds = []
                all_labels = []
                for it, inputs in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc='Validation'):
                    with torch.no_grad():
                        inputs = to_device(inputs, self.device)
                        labels = inputs.pop('labels')
                        outputs = self.model(**inputs).logits

                        all_preds.append(outputs.argmax(dim=-1).flatten())
                        all_labels.append(labels.flatten())
                all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
                all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
                all_preds[all_labels == -1] = -1

                epoch_metric = self.metric(all_preds, all_labels)
                all_metrics.append(epoch_metric)
                if epoch_metric > best_metric:
                    best_metric = epoch_metric
                    torch.save(self.model, os.path.join(save_path, 'best_model.pth'))

                print(f'Validation. Epoch: {epoch}, Macro F1: {epoch_metric}')
                self.model.train()
        return losses, all_metrics
