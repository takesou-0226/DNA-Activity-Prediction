import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

df = pd.read_csv("kadai1_data.csv")

MaxLen = df["sequence"].str.len().max()
Base2Index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def onehot_encoding(s):
    res = np.zeros((4, MaxLen))
    for i in range(len(s)):
        res[Base2Index[s[i]], i] = 1
    return res

X = np.zeros((df.shape[0], 4, MaxLen), dtype=np.float32)
for i in range(df.shape[0]):
    X[i] = onehot_encoding(df["sequence"][i])

y = df['activity'].to_numpy()

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=seed)
np.set_printoptions(
    threshold=np.inf,
    linewidth=200
)

class DNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tesor(y)
    def __len__(self):
        return len(self.X)
    def __getid__(self, id):
        return self.X[id], self.y[id]

train_dataset = DNADataset(X_train, y_train)
test_dataset = DNADataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64
)

'''class CNNRegressor(nn.Module):
    def __init__(self):
'''