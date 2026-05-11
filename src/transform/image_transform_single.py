import argparse
import os
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageOps

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
LEVELS = ['light', 'medium', 'heavy']


def crop_img(img, pct):
    w, h = img.size
    dx, dy = int(w * pct), int(h * pct)
    return img.crop((dx, dy, max(dx + 1, w - dx), max(dy + 1, h - dy)))


def resize_img(img, pct):
    w, h = img.size
    return img.resize((max(1, int(w * (1 - pct))), max(1, int(h * (1 - pct)))), Image.LANCZOS)


def add_border(img, pct):
    border = int(img.width * pct)
    return ImageOps.expand(img.convert('RGB'), border=border, fill='black')


def insert_on_canvas(img):
    img = img.convert('RGB')
    canvas = Image.new('RGB', (int(img.width * 1.4), int(img.height * 1.4)), (245, 245, 245))
    x = (canvas.width - img.width) // 2
    y = (canvas.height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def concatenate_with_self(img, direction='horizontal'):
    img = img.convert('RGB')
    if direction == 'vertical':
        canvas = Image.new('RGB', (img.width, img.height * 2), 'white')
        canvas.paste(img, (0, 0))
        canvas.paste(img, (0, img.height))
    else:
        canvas = Image.new('RGB', (img.width * 2, img.height), 'white')
        canvas.paste(img, (0, 0))
        canvas.paste(img, (img.width, 0))
    return canvas


TRANSFORMATIONS = {
    'crop': {
        'light': lambda img: crop_img(img, 0.10),
        'medium': lambda img: crop_img(img, 0.20),
        'heavy': lambda img: crop_img(img, 0.30),
    },
    'resolution': {
        'light': lambda img: resize_img(img, 0.10),
        'medium': lambda img: resize_img(img, 0.20),
        'heavy': lambda img: resize_img(img, 0.30),
    },
    'rotation': {
        'light': lambda img: img.rotate(90, expand=True),
        'medium': lambda img: img.rotate(270, expand=True),
    },
    'flip': {
        'light': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        'medium': lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
        'heavy': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
    },
    'border': {
        'light': lambda img: add_border(img, 0.10),
        'medium': lambda img: add_border(img, 0.20),
        'heavy': lambda img: add_border(img, 0.30),
    },
    'grayscale': {
        'light': lambda img: ImageOps.grayscale(img),
    },
    'brightness': {
        'light': lambda img: ImageEnhance.Brightness(img).enhance(1.10),
        'medium': lambda img: ImageEnhance.Brightness(img).enhance(1.20),
        'heavy': lambda img: ImageEnhance.Brightness(img).enhance(1.30),
    },
    'contrast': {
        'light': lambda img: ImageEnhance.Contrast(img).enhance(1.10),
        'medium': lambda img: ImageEnhance.Contrast(img).enhance(1.20),
        'heavy': lambda img: ImageEnhance.Contrast(img).enhance(1.30),
    },
    'format': {
        'light': 'JPEG',
        'medium': 'PNG',
        'heavy': 'WEBP',
    },
    'insertion': {
        'light': lambda img: insert_on_canvas(img),
    },
    'concatenation': {
        'light': lambda img: concatenate_with_self(img, 'horizontal'),
        'medium': lambda img: concatenate_with_self(img, 'vertical'),
    },
}


def is_image(path):
    return Path(path).suffix.lower() in IMG_EXTS


def transform_single_dataset(extracted_root, output_root):
    extracted_root = Path(extracted_root)
    output_root = Path(output_root)

    for field_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        for paper_dir in sorted([p for p in field_dir.iterdir() if p.is_dir()]):
            fig_dir = paper_dir / 'Figure'
            if not fig_dir.is_dir():
                continue

            for src in fig_dir.rglob('*'):
                if not src.is_file() or not is_image(src):
                    continue
                rel = src.relative_to(fig_dir)

                try:
                    img = Image.open(src)
                except Exception as e:
                    print(f'이미지 로딩 실패: {src} ({e})')
                    continue

                for ttype, level_map in TRANSFORMATIONS.items():
                    for level, func in level_map.items():
                        out_dir = output_root / field_dir.name / paper_dir.name / level / ttype / 'Figure' / rel.parent
                        out_dir.mkdir(parents=True, exist_ok=True)
                        stem = rel.stem

                        if ttype == 'format':
                            fmt = func
                            out_path = out_dir / f'{stem}.{fmt.lower()}'
                            img.convert('RGB').save(out_path, fmt)
                        else:
                            out_path = out_dir / rel.name
                            transformed = func(img)
                            if transformed.mode not in ('RGB', 'L'):
                                transformed = transformed.convert('RGB')
                            transformed.save(out_path)
    print(f'✅ 단일 변형 저장 완료 → {output_root}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_root', default='test_data_set/Extracted')
    parser.add_argument('--output_root', default='test_data_set/Transformed_single')
    args = parser.parse_args()
    transform_single_dataset(args.extracted_root, args.output_root)


if __name__ == '__main__':
    main()
