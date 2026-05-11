import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def analyze_by_grade(result_csv, pred_col, output_path='outputs/analysis/metrics_by_grade.csv'):
    df = pd.read_csv(result_csv)
    rows = []
    for grade, g in df.groupby('transform_grade'):
        y_true = g['label'].astype(int)
        y_pred = g[pred_col].astype(int)
        rows.append({
            'transform_grade': grade,
            'count': len(g),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        })
    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f'✅ 등급별 분석 저장 완료: {output_path}')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_csv', required=True)
    parser.add_argument('--pred_col', required=True)
    parser.add_argument('--output', default='outputs/analysis/metrics_by_grade.csv')
    args = parser.parse_args()
    analyze_by_grade(args.result_csv, args.pred_col, args.output)


if __name__ == '__main__':
    main()
