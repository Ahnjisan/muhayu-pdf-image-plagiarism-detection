import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.preprocess import preprocess_for_A, preprocess_for_B


class CsvImagePairDataset(Dataset):
    def __init__(self, csv_path, mode='B', return_paths=False, return_meta=False):
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.return_paths = return_paths
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.mode == 'A':
            img1 = preprocess_for_A(row['original_path'])
            img2 = preprocess_for_A(row['transformed_path'])
        else:
            img1 = preprocess_for_B(row['original_path'])
            img2 = preprocess_for_B(row['transformed_path'])

        label = torch.tensor(row['label'], dtype=torch.float32)
        output = [img1, img2, label]

        if self.return_paths:
            output.extend([row['original_path'], row['transformed_path']])

        if self.return_meta:
            meta_cols = [c for c in ['field', 'paper', 'transform_type', 'transform_level', 'transform_grade', 'total_score', 'transform_combo'] if c in self.df.columns]
            output.append({c: row[c] for c in meta_cols})

        return tuple(output)
