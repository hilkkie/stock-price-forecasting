# %%
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(os.getcwd())
DATA_ROOT = ROOT / "data"

# %%
# class representing the data
class StockPrice(Dataset):
    
    def __init__(self, seq_length, test_length, train=True, scaler=None):
        # complete data set
        self.data = pd.read_csv(DATA_ROOT / "data.csv")[["Date", "Close"]].sort_values("Date")
        
        # transform the data
        if scaler is not None:
            self.data = self.data.assign(
                CloseScaled = lambda x: scaler.fit_transform(x["Close"].to_numpy().reshape(-1, 1))
            )
        else:
            self.data = self.data.assign(
                CloseScaled = lambda x: x["Close"].to_numpy().reshape(-1, 1)
            )
        self.scaler = scaler
   
        # divide data into training and testing sets in chronological order
        if train:
            data = self.data.iloc[:(len(self.data) - (seq_length+test_length)), :]
            
            # for training, split the data into sequences  
            input_temp, target_temp = [], []
            for i in range(len(data)-(seq_length+test_length+1)):
                input_temp.append(data.iloc[i:(i+seq_length), 2].values)
                target_temp.append(data.iloc[(i+seq_length):(i+seq_length+1), 2].values)
    
            self.inputs = torch.stack([torch.from_numpy(x) for x in input_temp]).float()
            self.targets = torch.stack([torch.from_numpy(x) for x in target_temp]).float()
            
        else:
            self.inputs = torch.from_numpy(
                self.data.iloc[(len(self.data)-(seq_length+test_length)):(len(self.data)-test_length), 2].values
            ).float()
            self.targets = torch.from_numpy(self.data.iloc[(len(self.data)-test_length):, 2].values).float()
            
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        sample = self.inputs[idx,:]
        label = self.targets[idx]
        return sample, label    
    
# %%
# create train-test split and initialize data loader for training
seq_length, test_length = 5, 5
batch_size = 10
data_scaler = MinMaxScaler(feature_range=(-1, 1))

training_data = StockPrice(seq_length, test_length, train=True, scaler=data_scaler)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

testing_data = StockPrice(seq_length, test_length, train=False, scaler=data_scaler)

# %%
# visualize the data with train-test split
df = training_data.data
df.loc[:, ["Train", "Test"]] = np.nan
n_train = len(df) - (seq_length+test_length)
df.iloc[:n_train, 3] = df.iloc[:n_train, 1]
df.iloc[n_train:, 4] = df.iloc[n_train:, 1]

fig, ax = plt.subplots(1, 1, figsize=(16,6))
df.plot(x="Date", y=["Train", "Test"], ax=ax, style=["-", "--"], grid=True)

plt.show()

# %%
# model using LSTM
class LSTM(nn.Module):
    def __init__(self, n_input=1, n_hidden=10, n_output=1):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(n_input, n_hidden)
        self.linear = nn.Linear(n_hidden, n_output)
        
    def forward(self, x, hidden, state):
        output, (hidden, _) = self.lstm(x, (hidden, state))
        y = self.linear(output)
        return y[-1], hidden
    
    def init_hidden(self, batch_size):
        # initialize hidden and cell states to zeros
        return torch.zeros([1, batch_size, self.n_hidden], dtype=torch.float), torch.zeros([1, batch_size, self.n_hidden], dtype=torch.float)
    
# %%
lstm = LSTM()

N_epochs = 15
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

start_time = time.time()
for n in range(N_epochs):
    
    losses = []
    
    for inputs, outputs in train_dataloader: 
        # format the input for LSTM: batch size is the second dimension
        x = inputs.T.view((seq_length, batch_size, 1))
        
        optimizer.zero_grad()
        hidden, state = lstm.init_hidden(batch_size)
        y, _ = lstm.forward(x, hidden, state)
        loss = F.mse_loss(y, outputs)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    print(f"On epoch {n+1}, the average loss is {np.mean(losses)}.")
        
end_time = time.time()
print(f"Total training time {end_time - start_time}")