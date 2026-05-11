import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms

from src.model.model import SiameseNetwork
from src.data.preprocess import preprocess_for_A, preprocess_for_B, preprocess_for_cosine
from src.baseline.similarity_metrics import cosine_score, ssim_score, phash_score


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def predict_plagiarism(original_path, transformed_path, checkpoint_path=None, model_threshold_path='outputs/thresholds/best_threshold.json', baseline_threshold_path='outputs/thresholds/baseline_thresholds.json', device='cuda'):
    device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
    results = {}

    # baseline
    baseline_thresholds = _load_json(baseline_threshold_path)
    if baseline_thresholds:
        to_pil = transforms.ToPILImage()
        cos = cosine_score(preprocess_for_cosine(original_path), preprocess_for_cosine(transformed_path))
        img1_a = to_pil(preprocess_for_A(original_path))
        img2_a = to_pil(preprocess_for_A(transformed_path))
        ss = ssim_score(img1_a, img2_a)
        ph = phash_score(img1_a, img2_a)
        for name, score in [('Cosine', cos), ('SSIM', ss), ('pHash', ph)]:
            th = baseline_thresholds[name]['threshold'] if isinstance(baseline_thresholds[name], dict) else baseline_thresholds[name]
            results[name] = {'score': float(score), 'threshold': float(th), 'is_plagiarized': bool(score > th)}

    # model
    model_info = _load_json(model_threshold_path)
    if model_info:
        ckpt = checkpoint_path or model_info.get('checkpoint_path')
        if ckpt and Path(ckpt).exists():
            model_type = model_info.get('model_type', 'contrastive')
            threshold = model_info['threshold']
            model = SiameseNetwork(backbone_type='resnet18', mode=model_type)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model = model.to(device).eval()
            img1 = preprocess_for_B(original_path).unsqueeze(0).to(device)
            img2 = preprocess_for_B(transformed_path).unsqueeze(0).to(device)
            with torch.no_grad():
                if model_type == 'contrastive':
                    emb1, emb2 = model(img1, img2, return_embeddings=True)
                    distance = torch.norm(emb1 - emb2, p=2, dim=1)
                    score = -distance.item()
                else:
                    logit = model(img1, img2)
                    score = torch.sigmoid(logit).item()
            results['Model'] = {'score': float(score), 'threshold': float(threshold), 'is_plagiarized': bool(score > threshold), 'model_type': model_type}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', required=True)
    parser.add_argument('--transformed_path', required=True)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = predict_plagiarism(args.original_path, args.transformed_path, args.checkpoint_path, device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
