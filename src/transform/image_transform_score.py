import argparse
import os
import random
import time
from pathlib import Path
from PIL import Image

from src.transform.image_transform_single import TRANSFORMATIONS, is_image

SCORE_MAP = {'light': 1, 'medium': 2, 'heavy': 3}


def get_grade(total_score):
    if 1 <= total_score <= 3:
        return 'Light'
    if 4 <= total_score <= 6:
        return 'Medium'
    return 'Heavy'


def is_valid_combination(selected):
    types = [name for name, _ in selected]
    levels = {name: level for name, level in selected}

    if all(SCORE_MAP.get(level, 1) == 1 for _, level in selected):
        return False
    if levels.get('format') == 'heavy' and 'crop' not in types:
        return False
    return True


def select_random_transforms(min_k=2, max_k=4):
    candidates = list(TRANSFORMATIONS.keys())
    while True:
        k = random.randint(min_k, max_k)
        picks = random.sample(candidates, k=k)
        selected = []
        total_score = 0
        for name in picks:
            level = random.choice(list(TRANSFORMATIONS[name].keys()))
            selected.append((name, level))
            total_score += SCORE_MAP.get(level, 1)
        if is_valid_combination(selected):
            return selected, total_score, get_grade(total_score)


def apply_selected_transforms(img, selected):
    fmt = None
    out = img
    for name, level in selected:
        func = TRANSFORMATIONS[name][level]
        if name == 'format':
            fmt = func
        else:
            out = func(out)
    return out, fmt


def process_paper(field_dir, paper_dir, output_root, min_k=2, max_k=4):
    fig_dir = paper_dir / 'Figure'
    if not fig_dir.is_dir():
        return None

    selected, total_score, grade = select_random_transforms(min_k, max_k)
    combo = '__'.join(f'{name}_{level}' for name, level in selected)
    timestamp = time.strftime('%Y%m%dT%H%M%S')
    run_name = f'run_{timestamp}_{grade}_{total_score}_{combo}'
    run_dir = output_root / field_dir.name / paper_dir.name / run_name
    out_fig_dir = run_dir / 'Figure'
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    for src in fig_dir.rglob('*'):
        if not src.is_file() or not is_image(src):
            continue
        rel = src.relative_to(fig_dir)
        dest_folder = out_fig_dir / rel.parent
        dest_folder.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(src)
        except Exception as e:
            print(f'이미지 로딩 실패: {src} ({e})')
            continue
        transformed, fmt = apply_selected_transforms(img, selected)
        if fmt:
            out_path = dest_folder / f'{rel.stem}.{fmt.lower()}'
            transformed.convert('RGB').save(out_path, fmt)
        else:
            out_path = dest_folder / rel.name
            transformed.save(out_path)

    meta = run_dir / 'metadata.txt'
    meta.write_text(
        f'transforms: {selected}\n'
        f'total_score: {total_score}\n'
        f'grade: {grade}\n'
        f'timestamp: {timestamp}\n',
        encoding='utf-8'
    )
    print(f'[{field_dir.name}/{paper_dir.name}] → {run_name}')
    return run_dir


def transform_score_dataset(extracted_root, output_root, min_k=2, max_k=4, seed=42):
    random.seed(seed)
    extracted_root = Path(extracted_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for field_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        for paper_dir in sorted([p for p in field_dir.iterdir() if p.is_dir()]):
            process_paper(field_dir, paper_dir, output_root, min_k, max_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_root', default='test_data_set/Extracted')
    parser.add_argument('--output_root', default='test_data_set/Transformed_score')
    parser.add_argument('--min_k', type=int, default=2)
    parser.add_argument('--max_k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    transform_score_dataset(args.extracted_root, args.output_root, args.min_k, args.max_k, args.seed)


if __name__ == '__main__':
    main()
