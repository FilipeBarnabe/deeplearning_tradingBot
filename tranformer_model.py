import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# the features are separated into sequences of 128 days, 4 price features and 1 volume feature
seq_len = 128

# during a single step the model receives 32 sequences
batch_size = 32

"""
Our transformer model will have an input size of (32, 128, 5)
32 -> batch sie
128 -> sequence length
5 -> number of features (Open, Low, High, Close)
"""

# Stock for the deep learning
ticker = "GOOGL"

# Importing dataset (use the function in data/data.py)
df = pd.read_csv(
    "data/data.csv", usecols=["Date", "High", "Low", "Open", "Close", "Volume"]
)

# Selection of columns
# df = df[["Date", "High", "Low", "Open", "Close", "Volume"]]

# Avoid dividing by 0
df["Volume"].replace(to_replace=0, method="ffill", inplace=True)

# Sort the values based on date
df.sort_values("Date", inplace=True)

print(df.tail())

df["Open"] = df["Open"].pct_change()  # Create arithmetic returns column
df["High"] = df["High"].pct_change()  # Create arithmetic returns column
df["Low"] = df["Low"].pct_change()  # Create arithmetic returns column
df["Close"] = df["Close"].pct_change()  # Create arithmetic returns column
df["Volume"] = df["Volume"].pct_change()

# Drop the rows with the NaN created by the percentage change
df.dropna(how="any", axis=0, inplace=True)

# Get the values to create the separation of the dataset
times = sorted(df.index.values)
last_10pct = sorted(df.index.values)[-int(0.1 * len(times))]
last_20pct = sorted(df.index.values)[-int(0.2 * len(times))]

# min-max price columns
min_return = min(
    df[(df.index < last_20pct)][["Open", "High", "Low", "Close"]].min(axis=0)
)
max_return = max(
    df[(df.index < last_20pct)][["Open", "High", "Low", "Close"]].max(axis=0)
)

# Min-max normalize price columns (0-1 range)
df["Open"] = (df["Open"] - min_return) / (max_return - min_return)
df["High"] = (df["High"] - min_return) / (max_return - min_return)
df["Low"] = (df["Low"] - min_return) / (max_return - min_return)
df["Close"] = (df["Close"] - min_return) / (max_return - min_return)

# min-max volume column
min_volume = df[(df.index < last_20pct)]["Volume"].min(axis=0)
max_volume = df[(df.index < last_20pct)]["Volume"].max(axis=0)

# Min-max normalize volume columns (0-1 range)
df["Volume"] = (df["Volume"] - min_volume) / (max_volume - min_volume)

df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
df_test = df[(df.index >= last_10pct)]

# Drop the date column from the splitted datasets
df_train.drop(columns=["Date"], inplace=True)
df_val.drop(columns=["Date"], inplace=True)
df_test.drop(columns=["Date"], inplace=True)

# Train data into arrays np.ndarray
train_data = df_train.values
val_data = df_val.values
test_data = df_test.values
print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")

print(df_train.head())


class TickerData(torch.utils.data.Dataset):
    def __init__(self, df, seq_len):

        self.inputs, self.targets = [], []

        for i in range(seq_len, len(df)):
            # Chunks of  data with a length of 128 df-rows
            self.inputs.append(df[i - seq_len : i])

            # Value of 4th column (Close Price) of df-row 128+1
            self.targets.append(df[:, 3][i])

        self.inputs, self.targets = np.array(self.inputs), np.array(self.targets)

        ## *** ##

    def __getitem__(self, idx):
        return {"inputs": self.inputs[idx], "targets": self.targets[idx]}

    def __len__(self):
        return min(len(self.inputs), len(self.targets))
