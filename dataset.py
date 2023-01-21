import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
class SP500Dataset(Dataset):
  def __init__(self, csv_path, mode = 'training'):
    super(SP500Dataset).__init__()
    data = pd.read_csv(csv_path, index_col = 0)
    self.symbols = data.index.to_list()
    self.scaler = MinMaxScaler()
    data= self.transform(data)

    
    self.X, self.y = self.split_period(data = data, mode = mode)
    

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def transform(self, data):

    data = data.values.astype(np.float32)

    data = self.scaler.fit_transform(data)
    return data

  def split_period(self, data, periods = 30, mode = 'training'):
    X, y = [], []
    if mode == 'predict':
      data = data[-(periods + 12): -12]

    size = data.shape[1]

    for i in range(size):

      end_i = i + periods

      if end_i == size:
        break;
      X.extend(data[:, i:end_i])
      y.extend(data[:, end_i])
    return X, y
