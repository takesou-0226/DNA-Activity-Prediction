import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

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
torch.manual_seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=seed)
np.set_printoptions(
    threshold=np.inf,
    linewidth=200
)

class DNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveMaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 1)
    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        x3 = self.gap(x3).flatten(1)
        x5 = self.gap(x5).flatten(1)
        x7 = self.gap(x7).flatten(1)

        x = torch.cat([x3, x5, x7], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiScaleCNN().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001)

epochs = 100
loss_his = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
    print(
        f"Epoch {epoch+1}/{epochs}  Loss={total_loss:.4f}"
    )
    loss_his.append(avg_loss)
predictions = []
answers = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)
        predictions.extend(pred.cpu().numpy())
        answers.extend(y_batch.cpu().numpy())

predictions = np.array(predictions).flatten()
answers = np.array(answers).flatten()

print(f"r2 = {r2_score(answers, predictions)}")

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), loss_his)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()
