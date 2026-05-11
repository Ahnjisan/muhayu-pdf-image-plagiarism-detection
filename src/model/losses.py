import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        label = label.float()
        positive_loss = label * distance.pow(2)
        negative_loss = (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2)
        return (positive_loss + negative_loss).mean()
