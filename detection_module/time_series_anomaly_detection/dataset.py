import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict
import os
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm

class MultiCSVTimeSeriesDataset(Dataset):
    def __init__(self, 
                 csv_files: List[Path],
                 window_size: int,
                 pred_len: int,
                 feature_cols: List[str] = None):
        """
        Dataset for time series data from multiple CSV files
        
        Args:
            csv_files (List[Path]): List of CSV file paths to use
            window_size (int): Length of input sequence
            pred_len (int): Length of prediction sequence
            feature_cols (List[str]): List of column names to use as features
        """
        self.csv_files = csv_files
        self.window_size = window_size
        self.pred_len = pred_len

        # Initialize file data cache
        self.file_data = {}
        self.current_file_idx = 0
        self.current_sequence_idx = 0
        self.current_file = None
        self.current_data = None
        
        # Shuffle file order
        random.shuffle(self.csv_files)

        total_sequences = 0
        print("Calculating total sequences across all files...")
        for file_path in tqdm(self.csv_files):
            df = pd.read_csv(file_path)
            self.feature_cols = list(df.keys())[1:]
            file_length = len(df)
            sequences_in_file = file_length - self.window_size - self.pred_len + 1
            if sequences_in_file > 0:
                total_sequences += sequences_in_file
        self.total_sequences = total_sequences

        
        # Load first file
        self._load_next_file()
        print("Initializing dataset, loaded first file...")

        
    def _load_next_file(self):
        """Load the next CSV file into memory"""
        if self.current_file_idx < len(self.csv_files):
            self.current_file = self.csv_files[self.current_file_idx]
            # print(f"Loading file: {self.current_file}")
            
            # Read the CSV file
            df = pd.read_csv(self.current_file)
            if self.feature_cols:
                df = df[self.feature_cols]
            
            self.current_data = df.values
            self.current_sequence_idx = 0
            self.current_file_idx += 1
        else:
            # Reset to beginning of file list and reshuffle
            self.current_file_idx = 0
            random.shuffle(self.csv_files)
            self._load_next_file()
    
    def __len__(self) -> int:
        """Return total number of possible sequences across all files"""
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence from the current file, moving to next file when current is exhausted
        
        Args:
            idx (int): Dataset index (not used, maintained for compatibility)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequence and target sequence
        """
        # Check if we need to move to next file

        if self.current_sequence_idx + self.window_size + self.pred_len > len(self.current_data):
            # print(f"End of file {self.current_file} with size of {len(self.current_data)}, moving to next file.")
            self._load_next_file()
        
        # Extract sequence from current file
        sequence = self.current_data[
            self.current_sequence_idx:self.current_sequence_idx + self.window_size + self.pred_len
        ]
        
        # Split into input and target
        x = sequence[:self.window_size]
        y = sequence[self.window_size:]
        if y.shape[0] == 0:
            raise IndexError(f"End of file reached for {self.current_file} at sequence index {self.current_sequence_idx}.")
        # Move to next sequence
        self.current_sequence_idx += 1
        # Convert to tensors
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return x, y

def split_csv_files(data_dir: str, train_ratio: float = 0.7) -> Tuple[List[Path], List[Path]]:
    """
    Split CSV files into training and validation sets
    
    Args:
        data_dir (str): Directory containing CSV files
        train_ratio (float): Ratio of files to use for training
        
    Returns:
        Tuple[List[Path], List[Path]]: Lists of training and validation file paths
    """
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob('*.csv'))
    
    if not all_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    # Shuffle files
    random.shuffle(all_files)
    
    # Split files
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Found {len(all_files)} CSV files")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    
    return train_files, val_files

def create_dataloader(
    csv_files: List[Path],
    window_size: int,
    pred_len: int,
    batch_size: int,
    feature_cols: List[str] = None,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for time series data from multiple CSV files
    
    Args:
        csv_files (List[Path]): List of CSV file paths to use
        window_size (int): Length of input sequence
        pred_len (int): Length of prediction sequence
        batch_size (int): Batch size
        feature_cols (List[str]): List of column names to use as features
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: DataLoader for the time series data
    """
    dataset = MultiCSVTimeSeriesDataset(
        csv_files=csv_files,
        window_size=window_size,
        pred_len=pred_len,
        feature_cols=feature_cols
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def split_data(
    data: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets
    
    Args:
        data (np.ndarray): Input time series data
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, validation, and test sets
    """
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data 