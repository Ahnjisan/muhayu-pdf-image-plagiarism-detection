import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_curve
from skimage.metrics import structural_similarity as ssim
import imagehash
from src.data.preprocess import preprocess_for_A, preprocess_for_cosine
from torchvision import transforms


def cosine_score(t1, t2):
    v1, v2 = t1.flatten(), t2.flatten()
    return float(torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8))


def ssim_score(img1, img2):
    arr1 = np.array(img1.convert('L'))
    arr2 = np.array(img2.convert('L'))
    return float(ssim(arr1, arr2))


def phash_score(img1, img2):
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    return float(1 - (hash1 - hash2) / (len(hash1.hash) ** 2))


def find_best_threshold_by_roc(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(scores)) <= 1:
        raise ValueError('score가 모두 동일하여 threshold를 계산할 수 없습니다.')
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid = ~np.isinf(thresholds)
    fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])


def compute_similarity_scores(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    to_pil = transforms.ToPILImage()
    y_true, cosine_scores, ssim_scores, phash_scores = [], [], [], []

    for _, row in df.iterrows():
        p1, p2 = row['original_path'], row['transformed_path']
        cos1 = preprocess_for_cosine(p1)
        cos2 = preprocess_for_cosine(p2)
        cosine_scores.append(cosine_score(cos1, cos2))

        a1 = to_pil(preprocess_for_A(p1))
        a2 = to_pil(preprocess_for_A(p2))
        ssim_scores.append(ssim_score(a1, a2))
        phash_scores.append(phash_score(a1, a2))
        y_true.append(int(row['label']))

    return {'y_true': y_true, 'Cosine': cosine_scores, 'SSIM': ssim_scores, 'pHash': phash_scores}
