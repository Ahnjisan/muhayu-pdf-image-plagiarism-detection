import argparse
from pathlib import Path
import pandas as pd


def summarize_csv(csv_path):
    df = pd.read_csv(csv_path)
    summaries = {}
    summaries['label_counts'] = df['label'].value_counts().rename_axis('label').reset_index(name='count')
    if 'transform_type' in df.columns:
        summaries['transform_type_label_counts'] = df.groupby(['transform_type', 'label']).size().reset_index(name='count')
    if 'transform_grade' in df.columns:
        summaries['grade_label_counts'] = df.groupby(['transform_grade', 'label']).size().reset_index(name='count')
    return summaries


def save_stats(csv_dir='test_data_set/dataset_split', output_path='outputs/analysis/dataset_stats.xlsx'):
    csv_dir = Path(csv_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for split in ['train', 'valid', 'test']:
            csv_path = csv_dir / f'{split}.csv'
            if not csv_path.exists():
                continue
            summaries = summarize_csv(csv_path)
            for name, df in summaries.items():
                df.to_excel(writer, sheet_name=f'{split}_{name}'[:31], index=False)
    print(f'✅ 데이터셋 통계 저장 완료: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', default='test_data_set/dataset_split')
    parser.add_argument('--output_path', default='outputs/analysis/dataset_stats.xlsx')
    args = parser.parse_args()
    save_stats(args.csv_dir, args.output_path)


if __name__ == '__main__':
    main()
