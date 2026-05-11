import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.data.loader import get_dataloaders
from src.model.model import SiameseNetwork
from src.model.losses import ContrastiveLoss
from src.model.threshold import find_best_threshold_by_youden


def train(model, train_loader, valid_loader, epochs=5, lr=1e-3, device='cuda', loss_type='contrastive', checkpoint_dir='outputs/checkpoints', threshold_path='outputs/thresholds/best_threshold.json'):
    device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
    model = model.to(device)
    criterion = ContrastiveLoss(margin=1.0) if loss_type == 'contrastive' else nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(threshold_path).parent.mkdir(parents=True, exist_ok=True)

    best_acc = -1
    best_ckpt = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for img1, img2, label in tqdm(train_loader, desc=f'[Epoch {epoch}] Training'):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()

            if loss_type == 'contrastive':
                emb1, emb2 = model(img1, img2, return_embeddings=True)
                distance = torch.norm(emb1 - emb2, p=2, dim=1)
                loss = criterion(distance, label)
            else:
                logit = model(img1, img2)
                loss = criterion(logit, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'✅ Epoch {epoch}: Train Loss = {total_loss / max(1, len(train_loader)):.4f}')

        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for img1, img2, label in valid_loader:
                img1, img2 = img1.to(device), img2.to(device)
                if loss_type == 'contrastive':
                    emb1, emb2 = model(img1, img2, return_embeddings=True)
                    distance = torch.norm(emb1 - emb2, p=2, dim=1)
                    score = -distance
                else:
                    logit = model(img1, img2)
                    score = torch.sigmoid(logit)
                y_score.extend(score.cpu().numpy().tolist())
                y_true.extend(label.cpu().numpy().tolist())

        print(f'🧮 y_true contains {len(y_true)} labels')
        print('📊 Label counts:', Counter(y_true))
        best_thresh, best_tpr, best_fpr = find_best_threshold_by_youden(y_true, y_score)
        preds = [1 if s > best_thresh else 0 for s in y_score]
        acc = accuracy_score(y_true, preds)
        print(f'⭐ Best Threshold: {best_thresh:.4f} | TPR: {best_tpr:.4f}, FPR: {best_fpr:.4f}')
        print(f'🧪 Validation Accuracy: {acc:.4f}')

        ckpt_path = checkpoint_dir / f'checkpoint_epoch{epoch}_{loss_type}.pth'
        torch.save(model.state_dict(), ckpt_path)
        if acc > best_acc:
            best_acc = acc
            best_ckpt = str(ckpt_path)
            metadata = {
                'model_type': loss_type,
                'score_type': 'negative_distance' if loss_type == 'contrastive' else 'probability',
                'threshold': float(best_thresh),
                'tpr': float(best_tpr),
                'fpr': float(best_fpr),
                'accuracy': float(acc),
                'checkpoint_path': best_ckpt,
            }
            Path(threshold_path).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(f'✅ best threshold 저장: {threshold_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', default='test_data_set/dataset_split')
    parser.add_argument('--loss_type', choices=['bce', 'contrastive'], default='contrastive')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    (_, _), (train_loader_B, valid_loader_B) = get_dataloaders(args.csv_dir, args.batch_size, args.num_workers)
    model = SiameseNetwork(backbone_type='resnet18', mode=args.loss_type)
    train(model, train_loader_B, valid_loader_B, epochs=args.epochs, lr=args.lr, device=args.device, loss_type=args.loss_type)


if __name__ == '__main__':
    main()
