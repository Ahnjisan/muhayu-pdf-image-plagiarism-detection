import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import get_test_loader
from src.model.model import SiameseNetwork


def evaluate(model, loader, threshold_info, device='cuda', output_csv='outputs/predictions/model_results.csv', figure_path='outputs/figures/confusion_matrix.png'):
    device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
    model = model.to(device).eval()
    threshold = threshold_info['threshold']
    model_type = threshold_info.get('model_type', 'contrastive')
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for img1, img2, label in tqdm(loader, desc='🧪 Evaluating'):
            img1, img2 = img1.to(device), img2.to(device)
            if model_type == 'contrastive':
                emb1, emb2 = model(img1, img2, return_embeddings=True)
                distance = torch.norm(emb1 - emb2, p=2, dim=1)
                score = -distance
            else:
                logit = model(img1, img2)
                score = torch.sigmoid(logit)
            y_true.extend(label.numpy().tolist())
            y_score.extend(score.cpu().numpy().tolist())
            y_pred.extend((score > threshold).cpu().numpy().astype(int).tolist())

    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_score),
        'threshold': threshold,
        'model_type': model_type,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    print(json.dumps(results, indent=2))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(output_csv, index=False)

    cm = confusion_matrix(y_true, y_pred)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Plag', 'Plag'], yticklabels=['Non-Plag', 'Plag'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', default='test_data_set/dataset_split')
    parser.add_argument('--threshold_path', default='outputs/thresholds/best_threshold.json')
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    _, test_loader_B = get_test_loader(args.csv_dir, args.batch_size, args.num_workers)
    threshold_info = json.loads(Path(args.threshold_path).read_text(encoding='utf-8'))
    checkpoint_path = args.checkpoint_path or threshold_info.get('checkpoint_path')
    model_type = threshold_info.get('model_type', 'contrastive')
    model = SiameseNetwork(backbone_type='resnet18', mode=model_type)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device if torch.cuda.is_available() else 'cpu'))
    evaluate(model, test_loader_B, threshold_info, args.device)


if __name__ == '__main__':
    main()
