import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


df = pd.read_csv("kadai1_data.csv")

df["length"] = df["sequence"].str.len()

print(df[["length", "activity"]].corr())