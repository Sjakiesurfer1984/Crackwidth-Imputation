import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# The Dataset is the low-level data access layer. Its primary job is to tell
# PyTorch how to get one single data sample from your entire collection.
class CrackDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_cols: list, target_col: str, window_size: int):
        self.df = df
        self.input_cols = input_cols
        self.target_col = target_col
        self.window_size = window_size
        
        # Convert specified columns to a NumPy array for efficient slicing
        self.data_array = self.df[[*input_cols, target_col]].to_numpy()
    
    def __len__(self):
        return len(self.df) - self.window_size
    
    def __getitem__(self, idx: int):
        start_idx = idx
        end_idx = idx + self.window_size
        
        # Slicing the input features for the window
        # The input is the full window's features
        X_window = self.data_array[start_idx:end_idx, :-1]
        
        # The target is the value at the end of the window
        y_target = self.data_array[end_idx, -1]

        # Convert to PyTorch tensors and return
        return torch.tensor(X_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)
    

# The DataModule is the high-level data management layer. Its job is to handle the entire data pipeline and 
# make it easy to plug into a PyTorch Lightning Trainer. It orchestrates the Dataset and DataLoader classes.
class CrackDataModule(pl.LightningDataModule):
    def __init__(self, data_df: pd.DataFrame, input_cols: list, target_col: str, window_size: int = 12, batch_size: int = 32):
        super().__init__()
        # Data is passed directly as a DataFrame, no need to load from a file
        self.data_df = data_df
        self.input_cols = input_cols
        self.target_col = target_col
        self.window_size = window_size
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        # This hook is no longer needed as the data is already in a DataFrame.
        # It's kept here as a placeholder for PyTorch Lightning's API.
        pass

    def setup(self, stage=None):
        # This hook is called on every GPU and every time after prepare_data.
        # We'll use it to create our train, validation, and test splits.
        
        # Split the data chronologically (this is important for time series!)
        train_val_test_split = [0.8, 0.1, 0.1]
        
        n_rows = len(self.data_df)
        n_train = int(n_rows * train_val_test_split[0])
        n_val = int(n_rows * train_val_test_split[1])

        # Create the splits
        train_df = self.data_df.iloc[:n_train].copy()
        val_df = self.data_df.iloc[n_train:n_train + n_val].copy()
        test_df = self.data_df.iloc[n_train + n_val:].copy()
        
        # Create our PyTorch Datasets
        self.train_data = CrackDataset(train_df, self.input_cols, self.target_col, self.window_size)
        self.val_data = CrackDataset(val_df, self.input_cols, self.target_col, self.window_size)
        self.test_data = CrackDataset(test_df, self.input_cols, self.target_col, self.window_size)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


