import argparse
import json
from pathlib import Path
import pandas as pd


def compare(summary_json='outputs/predictions/baseline_summary.json', model_csv='outputs/predictions/model_results.csv', output='outputs/analysis/model_comparison.csv'):
    rows = []
    if Path(summary_json).exists():
        data = json.loads(Path(summary_json).read_text(encoding='utf-8'))
        for method, metrics in data.items():
            row = {'method': method}
            row.update({k: v for k, v in metrics.items() if k != 'confusion_matrix'})
            rows.append(row)
    if Path(model_csv).exists():
        df = pd.read_csv(model_csv)
        if not df.empty:
            row = df.iloc[0].to_dict()
            row['method'] = f"Siamese_{row.get('model_type', 'model')}"
            rows.append(row)
    out = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f'✅ 모델 비교표 저장 완료: {output}')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_json', default='outputs/predictions/baseline_summary.json')
    parser.add_argument('--model_csv', default='outputs/predictions/model_results.csv')
    parser.add_argument('--output', default='outputs/analysis/model_comparison.csv')
    args = parser.parse_args()
    compare(args.summary_json, args.model_csv, args.output)


if __name__ == '__main__':
    main()
