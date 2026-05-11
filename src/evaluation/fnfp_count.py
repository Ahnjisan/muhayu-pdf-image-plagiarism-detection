import argparse
from pathlib import Path
import pandas as pd


def count_fnfp(result_files, output_path='outputs/analysis/fnfp_counts.csv'):
    frames = []
    for method, path in result_files.items():
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        pred_col = next((c for c in df.columns if c.endswith('_pred')), None)
        if pred_col is None and 'prediction' in df.columns:
            pred_col = 'prediction'
        if pred_col is None:
            continue
        df['method'] = method
        df['pred'] = df[pred_col].astype(int)
        frames.append(df)
    if not frames:
        raise ValueError('분석할 result csv가 없습니다.')

    all_df = pd.concat(frames, ignore_index=True)
    group_col = 'transform_combo' if 'transform_combo' in all_df.columns else 'transform_type'
    rows = []
    for (method, transform), g in all_df.groupby(['method', group_col]):
        label = g['label'].astype(int)
        pred = g['pred'].astype(int)
        fp = int(((label == 0) & (pred == 1)).sum())
        fn = int(((label == 1) & (pred == 0)).sum())
        neg = int((label == 0).sum())
        pos = int((label == 1).sum())
        rows.append({
            'method': method,
            group_col: transform,
            'total_positive': pos,
            'total_negative': neg,
            'fp_count': fp,
            'fn_count': fn,
            'fp_rate': fp / neg if neg else 0,
            'fn_rate': fn / pos if pos else 0,
        })
    out = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f'✅ FP/FN 분석 저장 완료: {output_path}')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cosine', default='outputs/predictions/cosine_results.csv')
    parser.add_argument('--ssim', default='outputs/predictions/ssim_results.csv')
    parser.add_argument('--phash', default='outputs/predictions/phash_results.csv')
    parser.add_argument('--model', default='outputs/predictions/model_pair_results.csv')
    parser.add_argument('--output', default='outputs/analysis/fnfp_counts.csv')
    args = parser.parse_args()
    count_fnfp({'Cosine': args.cosine, 'SSIM': args.ssim, 'pHash': args.phash, 'Model': args.model}, args.output)


if __name__ == '__main__':
    main()
