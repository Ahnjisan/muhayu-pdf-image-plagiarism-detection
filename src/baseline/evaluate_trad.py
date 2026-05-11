import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.baseline.similarity_metrics import compute_similarity_scores


def evaluate_traditional(csv_path, threshold=0.75):
    score_dict = compute_similarity_scores(csv_path)
    y_true = score_dict['y_true']
    results = {}
    for method in ['Cosine', 'SSIM', 'pHash']:
        scores = score_dict[method]
        y_pred = [1 if s > threshold else 0 for s in scores]
        results[method] = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, scores),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='test_data_set/dataset_split/test.csv')
    parser.add_argument('--threshold', type=float, default=0.75)
    args = parser.parse_args()
    results = evaluate_traditional(args.csv, args.threshold)
    for method, metrics in results.items():
        print(f'\n[{method}]')
        for k, v in metrics.items():
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
