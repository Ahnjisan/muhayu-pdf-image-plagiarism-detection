import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseNetwork(nn.Module):
    def __init__(self, backbone_type='resnet18', embedding_dim=512, mode='contrastive'):
        super().__init__()
        self.mode = mode

        if backbone_type == 'resnet18':
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone_type == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            num_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            raise ValueError(f'지원하지 않는 backbone_type: {backbone_type}')

        self.embedding = nn.Linear(num_features, embedding_dim)
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        if self.mode == 'contrastive':
            x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2, return_embeddings=False):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        if return_embeddings:
            return f1, f2
        diff = torch.abs(f1 - f2)
        logit = self.classifier(diff).squeeze(1)
        return logit
