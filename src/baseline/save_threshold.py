import argparse
import json
from pathlib import Path
from src.baseline.similarity_metrics import compute_similarity_scores, find_best_threshold_by_roc


def save_thresholds(valid_csv, output_path='outputs/thresholds/baseline_thresholds.json'):
    scores = compute_similarity_scores(valid_csv)
    thresholds = {}
    for method in ['Cosine', 'SSIM', 'pHash']:
        best, j, tpr, fpr = find_best_threshold_by_roc(scores['y_true'], scores[method])
        thresholds[method] = {'threshold': round(float(best), 6), 'youden_j': j, 'tpr': tpr, 'fpr': fpr}
        print(f'[{method}] threshold={best:.4f}, J={j:.4f}')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(thresholds, indent=2), encoding='utf-8')
    print(f'✅ 저장 완료: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='test_data_set/dataset_split/valid.csv')
    parser.add_argument('--output', default='outputs/thresholds/baseline_thresholds.json')
    args = parser.parse_args()
    save_thresholds(args.csv, args.output)


if __name__ == '__main__':
    main()
