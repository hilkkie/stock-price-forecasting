# %%
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
            data = self.data.iloc[:(len(self.data) - test_length), :]
            
            # for training, split the data into sequences  
            input_temp, target_temp = [], []
            for i in range(len(data)-(seq_length+1)):
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
test_length = 5 # forecasting period
seq_length_small, seq_length_medium, seq_length_large = 5, 10, 25
batch_size = 10
data_scaler = MinMaxScaler(feature_range=(-1, 1))

# create three different data sets with varying sequence lengths
training_data_small = StockPrice(seq_length_small, test_length, train=True, scaler=data_scaler)
train_dataloader_small = DataLoader(training_data_small, batch_size=batch_size, shuffle=True, drop_last=True)
testing_data_small = StockPrice(seq_length_small, test_length, train=False, scaler=data_scaler)

training_data_medium = StockPrice(seq_length_medium, test_length, train=True, scaler=data_scaler)
train_dataloader_medium = DataLoader(training_data_medium, batch_size=batch_size, shuffle=True, drop_last=True)
testing_data_medium = StockPrice(seq_length_medium, test_length, train=False, scaler=data_scaler)

training_data_large = StockPrice(seq_length_large, test_length, train=True, scaler=data_scaler)
train_dataloader_large = DataLoader(training_data_large, batch_size=batch_size, shuffle=True, drop_last=True)
testing_data_large = StockPrice(seq_length_large, test_length, train=False, scaler=data_scaler)

# %%
# visualize the data with train-test split
df = training_data_small.data.copy().assign(Date = lambda x: pd.to_datetime(x["Date"], format="%Y-%m-%d"))
df.loc[:, ["Train", "Test"]] = np.nan
n_train = len(df) - test_length
df.iloc[:n_train, 3] = df.iloc[:n_train, 1]
df.iloc[n_train:, 4] = df.iloc[n_train:, 1]

fig, ax = plt.subplots(1, 1, figsize=(16,6))

df.plot(x="Date", y=["Train", "Test"], ax=ax, style=["-", "--"], grid=True)

# set major ticks every year, minor tick every month
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
# format x-axis ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for label in ax.get_xticklabels(which="major"):
    label.set(rotation=30)

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
    
    def forecast(self, x, period_length, scaler=None):
        # forecast a sequence of length period_length
        y_fct = torch.zeros(period_length)
        
        self.eval()
        with torch.no_grad():
            for n in range(period_length):
                hidden, state = self.init_hidden(1)
                y, _ = self.forward(x.reshape((len(x), 1, 1)), hidden, state)
        
                # update the state
                x = torch.cat((x, y.flatten()))[1:]
                y_fct[n] = y.flatten()
                
        # scale the output if transformation is given
        if scaler is not None:
            y_fct = torch.from_numpy(scaler.inverse_transform(y_fct.numpy()))
            
        return y_fct
    
    def calculate_loss(self, test_data, scale_output=False):
        # calculate loss over test data
        x, targets = test_data.inputs, test_data.targets
        scaler = test_data.scaler if scale_output else None
        y = self.forecast(x, len(targets), scaler=scaler)
                
        return F.mse_loss(y, targets)
    
    def train_model(self, N_epochs, learning_rate, train_dl, seq_length, batch_size, test_data):
        # initialize optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        start_time = time.time()
        for n in range(N_epochs):
    
            losses = []
            test_losses = []
    
            for inputs, outputs in train_dl: 
                # format the input for LSTM: batch size is the second dimension
                x = inputs.T.view((seq_length, batch_size, 1))
        
                # perform single step of optimization
                optimizer.zero_grad()
                hidden, state = self.init_hidden(batch_size)
                y, _ = self.forward(x, hidden, state)
                loss = F.mse_loss(y, outputs)
                loss.backward()
                optimizer.step()
        
                # store training and test losses for monitoring
                losses.append(loss.item())
                test_losses.append(self.calculate_loss(test_data))
                self.train()
    
            # monitor progress
            print(f"On epoch {n+1}, the average trainig loss is {np.mean(losses):.6f} and average test error is {np.mean(test_losses):.6f}.")
        
        end_time = time.time()
        print(f"Total training time {end_time - start_time}")
    
# %%
# create a model and train it using small training data
lstm_small = LSTM(n_hidden=30)

lstm_small.train_model(20, 0.001, train_dataloader_small, seq_length_small, batch_size, testing_data_small)

# add LSTM prediction to the constructed data frame
test_input, test_targets = testing_data_small.inputs, testing_data_small.targets
ypred = lstm_small.forecast(test_input, len(test_targets), scaler=testing_data_small.scaler)
ypred = ypred.numpy().flatten()

df.loc[:, "LSTM, small"] = np.nan
df.iloc[(len(df) - len(ypred)):, 5] = ypred

# %%
# create a model and train it using medium training data
lstm_medium = LSTM(n_hidden=30)

lstm_medium.train_model(30, 0.001, train_dataloader_medium, seq_length_medium, batch_size, testing_data_medium)

# add LSTM prediction to the constructed data frame
test_input, test_targets = testing_data_medium.inputs, testing_data_medium.targets
ypred = lstm_medium.forecast(test_input, len(test_targets), scaler=testing_data_medium.scaler)
ypred = ypred.numpy().flatten()

df.loc[:, "LSTM, medium"] = np.nan
df.iloc[(len(df) - len(ypred)):, 6] = ypred

# %%
# create a model and train it using training data
lstm_large = LSTM(n_hidden=50)

lstm_large.train_model(30, 0.001, train_dataloader_large, seq_length_large, batch_size, testing_data_large)

# add LSTM prediction to the constructed data frame
test_input, test_targets = testing_data_large.inputs, testing_data_large.targets
ypred = lstm_large.forecast(test_input, len(test_targets), scaler=testing_data_large.scaler)
ypred = ypred.numpy().flatten()

df.loc[:, "LSTM, large"] = np.nan
df.iloc[(len(df) - len(ypred)):, 7] = ypred

# %%
# plot the forecasts
fig, ax = plt.subplots(1, 1, figsize=(4,4))

df.iloc[-20:, :].plot(
    x="Date",
    y=["Train", "Test", "LSTM, small", "LSTM, medium", "LSTM, large"],
    ax=ax,
    # style=["-", "--", ":"],
    grid=True
)

ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_minor_locator(mdates.DayLocator())

plt.show()