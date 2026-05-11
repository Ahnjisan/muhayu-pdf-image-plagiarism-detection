import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.baseline.similarity_metrics import compute_similarity_scores
from src.utils.utils_io import trim_path


def load_thresholds(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {k: (v['threshold'] if isinstance(v, dict) else v) for k, v in data.items()}


def generate_baseline_results(csv_path, threshold_path='outputs/thresholds/baseline_thresholds.json', output_dir='outputs/predictions'):
    df = pd.read_csv(csv_path)
    score_dict = compute_similarity_scores(csv_path)
    thresholds = load_thresholds(threshold_path)
    y_true = score_dict['y_true']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for method in ['Cosine', 'SSIM', 'pHash']:
        scores = score_dict[method]
        threshold = thresholds[method]
        preds = [1 if s > threshold else 0 for s in scores]
        out = df.copy()
        out['original_path_short'] = out['original_path'].apply(trim_path)
        out['transformed_path_short'] = out['transformed_path'].apply(trim_path)
        out[f'{method.lower()}_score'] = scores
        out[f'{method.lower()}_pred'] = preds
        out.to_csv(output_dir / f'{method.lower()}_results.csv', index=False)
        summary[method] = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall': recall_score(y_true, preds, zero_division=0),
            'f1_score': f1_score(y_true, preds, zero_division=0),
            'roc_auc': roc_auc_score(y_true, scores),
            'confusion_matrix': confusion_matrix(y_true, preds).tolist(),
        }
    Path(output_dir / 'baseline_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='test_data_set/dataset_split/test.csv')
    parser.add_argument('--thresholds', default='outputs/thresholds/baseline_thresholds.json')
    parser.add_argument('--output_dir', default='outputs/predictions')
    args = parser.parse_args()
    summary = generate_baseline_results(args.csv, args.thresholds, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
