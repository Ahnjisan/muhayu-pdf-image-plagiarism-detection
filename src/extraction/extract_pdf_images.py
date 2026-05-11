import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pdf2image import convert_from_path

try:
    import layoutparser as lp
except ImportError:  # pragma: no cover
    lp = None

from src.utils.config import load_yaml


def build_model(config):
    if lp is None:
        raise ImportError('layoutparser가 설치되어 있지 않습니다. requirements.txt와 detectron2 설치를 확인하세요.')

    label_map_raw = config['publaynet']['label_map']
    label_map = {int(k): v for k, v in label_map_raw.items()}
    return lp.Detectron2LayoutModel(
        config_path=config['publaynet']['config_path'],
        model_path=config['publaynet']['model_path'],
        label_map=label_map,
        extra_config=['MODEL.ROI_HEADS.SCORE_THRESH_TEST', config['publaynet'].get('score_threshold', 0.8)],
        device=config['publaynet'].get('device', 'cuda:0'),
    )


def extract_figures_from_pdf(pdf_path, output_root, model, target_labels=('Figure',), dpi=300):
    pdf_path = Path(pdf_path)
    output_root = Path(output_root)
    paper_name = pdf_path.stem
    paper_dir = output_root / paper_name
    paper_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    saved = []
    print(f'📄 {pdf_path.name} 페이지 수: {len(pages)}')

    for page_num, img in enumerate(pages, start=1):
        img_np = np.array(img)
        layout = model.detect(img_np)
        print(f'🔎 page {page_num}: detected {len(layout)} blocks')

        for idx, block in enumerate(layout, start=1):
            label = block.type
            if label not in target_labels:
                continue

            x1, y1, x2, y2 = map(int, block.coordinates)
            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label_dir = paper_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            out_path = label_dir / f'{paper_name}_page{page_num}_{label}_{idx}.png'
            Image.fromarray(crop).save(out_path)
            saved.append(out_path)
            print(f'  저장: {out_path}')
    return saved


def extract_from_directory(pdf_root, output_root, model, target_labels=('Figure',), dpi=300):
    pdf_root = Path(pdf_root)
    output_root = Path(output_root)

    for field_dir in sorted([p for p in pdf_root.iterdir() if p.is_dir()]):
        field_output = output_root / field_dir.name
        field_output.mkdir(parents=True, exist_ok=True)
        for pdf_path in sorted(field_dir.glob('*.pdf')):
            extract_figures_from_pdf(pdf_path, field_output, model, target_labels, dpi)


def main(config_path='configs/extraction_config.yaml'):
    config = load_yaml(config_path)
    model = build_model(config)
    extract_from_directory(
        config['paths']['pdf_root'],
        config['paths']['output_root'],
        model,
        tuple(config.get('target_labels', ['Figure'])),
        int(config['publaynet'].get('dpi', 300)),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/extraction_config.yaml')
    args = parser.parse_args()
    main(args.config)
