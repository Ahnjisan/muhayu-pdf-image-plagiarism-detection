import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

from src.utils.utils_io import is_image_file, parse_score_run_name


def find_original_by_stem(fig_dir, target_stem):
    for p in fig_dir.rglob('*'):
        if p.is_file() and is_image_file(p) and p.stem == target_stem:
            return p
    return None


def collect_originals(extracted_dir):
    extracted_dir = Path(extracted_dir)
    originals = []
    for fig_dir in extracted_dir.glob('*/*/Figure'):
        field = fig_dir.parent.parent.name
        paper = fig_dir.parent.name
        for img in fig_dir.rglob('*'):
            if img.is_file() and is_image_file(img):
                originals.append({'path': img, 'field': field, 'paper': paper, 'stem': img.stem})
    return originals


def collect_single_positive_pairs(extracted_dir, transformed_single_dir):
    extracted_dir = Path(extracted_dir)
    root = Path(transformed_single_dir)
    pairs = []
    for img in root.rglob('*'):
        if not img.is_file() or not is_image_file(img):
            continue
        try:
            rel = img.relative_to(root)
            field, paper, level, ttype = rel.parts[0], rel.parts[1], rel.parts[2], rel.parts[3]
        except Exception:
            continue
        fig_dir = extracted_dir / field / paper / 'Figure'
        orig = find_original_by_stem(fig_dir, img.stem)
        if not orig:
            continue
        grade = level.capitalize()
        score = {'light': 1, 'medium': 2, 'heavy': 3}.get(level, 1)
        pairs.append({
            'original_path': str(orig), 'transformed_path': str(img), 'label': 1,
            'field': field, 'paper': paper, 'transform_type': ttype,
            'transform_level': level, 'transform_grade': grade,
            'total_score': score, 'transform_combo': f'{ttype}_{level}',
        })
    return pairs


def collect_score_positive_pairs(extracted_dir, transformed_score_dir):
    extracted_dir = Path(extracted_dir)
    root = Path(transformed_score_dir)
    pairs = []
    for run_dir in root.glob('*/*/run_*'):
        if not run_dir.is_dir():
            continue
        field = run_dir.parent.parent.name
        paper = run_dir.parent.name
        run_info = parse_score_run_name(run_dir.name)
        fig_dir = run_dir / 'Figure'
        orig_fig_dir = Path(extracted_dir) / field / paper / 'Figure'
        if not fig_dir.is_dir():
            continue
        for img in fig_dir.rglob('*'):
            if not img.is_file() or not is_image_file(img):
                continue
            orig = find_original_by_stem(orig_fig_dir, img.stem)
            if not orig:
                continue
            pairs.append({
                'original_path': str(orig), 'transformed_path': str(img), 'label': 1,
                'field': field, 'paper': paper, 'transform_type': 'score_combo',
                'transform_level': 'combo',
                'transform_grade': run_info['transform_grade'],
                'total_score': run_info['total_score'],
                'transform_combo': run_info['transform_combo'],
            })
    return pairs


def make_negative_pair(positive_pair, originals_by_field_paper, all_originals):
    # same transformed image, different original image => non-plagiarism pair
    field, paper = positive_pair['field'], positive_pair['paper']
    candidates = [o for o in all_originals if str(o['path']) != positive_pair['original_path']]
    if not candidates:
        return None
    # 가능하면 같은 논문이 아닌 원본을 우선 선택
    far_candidates = [o for o in candidates if not (o['field'] == field and o['paper'] == paper)]
    src = random.choice(far_candidates or candidates)
    neg = dict(positive_pair)
    neg['original_path'] = str(src['path'])
    neg['label'] = 0
    return neg


def balance_pairs_by_transform(positive_pairs, originals, total_pairs=30000, positive_ratio=0.5):
    pos_by_type = defaultdict(list)
    for p in positive_pairs:
        key = p['transform_combo'] if p['transform_type'] == 'score_combo' else f"{p['transform_type']}/{p['transform_level']}"
        pos_by_type[key].append(p)

    target_pos_total = int(total_pairs * positive_ratio)
    target_neg_total = total_pairs - target_pos_total
    types = list(pos_by_type.keys())
    if not types:
        raise ValueError('positive pair가 없습니다. Extracted/Transformed 경로를 확인하세요.')

    per_type_pos = max(1, target_pos_total // len(types))
    balanced = []
    originals_by_field_paper = defaultdict(list)
    for o in originals:
        originals_by_field_paper[(o['field'], o['paper'])].append(o)

    for key, vals in pos_by_type.items():
        sample_size = min(per_type_pos, len(vals))
        sampled_pos = random.sample(vals, sample_size) if len(vals) > sample_size else vals[:]
        balanced.extend(sampled_pos)
        for p in sampled_pos:
            neg = make_negative_pair(p, originals_by_field_paper, originals)
            if neg:
                balanced.append(neg)

    # total_pairs보다 많으면 줄이고, 적으면 가능한 positive에서 추가 생성
    random.shuffle(balanced)
    if len(balanced) > total_pairs:
        balanced = balanced[:total_pairs]
    return balanced


def split_pairs(pairs, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    return {
        'train': pairs[:train_end],
        'valid': pairs[train_end:valid_end],
        'test': pairs[valid_end:],
    }


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['original_path', 'transformed_path', 'label', 'field', 'paper', 'transform_type', 'transform_level', 'transform_grade', 'total_score', 'transform_combo']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def save_metadata(rows, path):
    c_label = Counter(str(r['label']) for r in rows)
    c_type = Counter(r.get('transform_type', 'unknown') for r in rows)
    c_grade = Counter(r.get('transform_grade', 'unknown') for r in rows)
    text = []
    text.append(f'total_pairs: {len(rows)}')
    text.append(f'label_counts: {dict(c_label)}')
    text.append(f'transform_type_counts: {dict(c_type)}')
    text.append(f'transform_grade_counts: {dict(c_grade)}')
    Path(path).write_text('\n'.join(text), encoding='utf-8')


def generate_dataset(extracted_dir, transformed_single_dir, transformed_score_dir, output_dir, total_pairs=30000, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    originals = collect_originals(extracted_dir)
    pos_single = collect_single_positive_pairs(extracted_dir, transformed_single_dir)
    pos_score = collect_score_positive_pairs(extracted_dir, transformed_score_dir)
    positive_pairs = pos_single + pos_score

    pairs = balance_pairs_by_transform(positive_pairs, originals, total_pairs=total_pairs, positive_ratio=0.5)
    splits = split_pairs(pairs, train_ratio, valid_ratio, seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        save_csv(rows, output_dir / f'{name}.csv')
        save_metadata(rows, output_dir / f'{name}_metadata.txt')
        print(f'✅ {name}: {len(rows)} pairs 저장')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_dir', default='test_data_set/Extracted')
    parser.add_argument('--transformed_single_dir', default='test_data_set/Transformed_single')
    parser.add_argument('--transformed_score_dir', default='test_data_set/Transformed_score')
    parser.add_argument('--output_dir', default='test_data_set/dataset_split')
    parser.add_argument('--total_pairs', type=int, default=30000)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_dataset(**vars(args))


if __name__ == '__main__':
    main()
