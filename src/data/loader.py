from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import CsvImagePairDataset


def get_dataloaders(csv_dir='test_data_set/dataset_split', batch_size=32, num_workers=4):
    csv_dir = Path(csv_dir)
    train_csv = csv_dir / 'train.csv'
    valid_csv = csv_dir / 'valid.csv'

    train_loader_A = DataLoader(CsvImagePairDataset(train_csv, mode='A'), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader_A = DataLoader(CsvImagePairDataset(valid_csv, mode='A'), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader_B = DataLoader(CsvImagePairDataset(train_csv, mode='B'), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader_B = DataLoader(CsvImagePairDataset(valid_csv, mode='B'), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return (train_loader_A, valid_loader_A), (train_loader_B, valid_loader_B)


def get_test_loader(csv_dir='test_data_set/dataset_split', batch_size=32, num_workers=4):
    test_csv = Path(csv_dir) / 'test.csv'
    test_loader_A = DataLoader(CsvImagePairDataset(test_csv, mode='A'), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader_B = DataLoader(CsvImagePairDataset(test_csv, mode='B'), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader_A, test_loader_B
