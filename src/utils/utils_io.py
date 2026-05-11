from pathlib import Path
import re


def trim_path(full_path):
    text = str(full_path).replace('\\', '/')
    for keyword in ['Extracted', 'Transformed_single', 'Transformed_score']:
        marker = f'{keyword}/'
        if marker in text:
            return text.split(marker, 1)[-1]
    return text


def extract_transform_type(transformed_path):
    text = str(transformed_path).replace('\\', '/')
    parts = text.split('/')

    if 'Transformed_score' in parts:
        run = next((p for p in parts if p.startswith('run_')), None)
        if run:
            # run_YYYYMMDDTHHMMSS_Medium_5_format_medium__brightness_heavy
            tokens = run.split('_')
            if len(tokens) >= 5:
                return '__'.join(tokens[4:])
            return run

    if 'Transformed_single' in parts:
        try:
            idx = parts.index('Transformed_single')
            # Transformed_single/field/paper/level/type/Figure/file
            level = parts[idx + 3]
            ttype = parts[idx + 4]
            return f'{ttype}/{level}'
        except Exception:
            return 'unknown'

    return 'none'


def parse_score_run_name(run_name):
    # run_20250520T063444_Medium_5_format_medium__brightness_heavy
    if not run_name.startswith('run_'):
        return {'transform_grade': 'unknown', 'total_score': None, 'transform_combo': run_name}
    m = re.match(r'^run_[^_]+_([^_]+)_([0-9]+)_(.+)$', run_name)
    if not m:
        return {'transform_grade': 'unknown', 'total_score': None, 'transform_combo': run_name}
    return {
        'transform_grade': m.group(1),
        'total_score': int(m.group(2)),
        'transform_combo': m.group(3),
    }


def is_image_file(path):
    return Path(path).suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
