import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("kadai1_data.csv")

MaxLen = df["sequence"].str.len().max()
Base2Index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def onehot_encoding(s):
    res = np.zeros((4, MaxLen))
    for i in range(len(s)):
        res[Base2Index[s[i]], i] = 1
    return res.flatten()

X = np.zeros([df.shape[0], MaxLen * 4])
for i in range(df.shape[0]):
    X[i, :] = onehot_encoding(df["sequence"][i])

y = df['activity'].to_numpy()

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=seed)

model = ExtraTreesRegressor()
param_grid = {'n_estimators':[10, 20, 50, 100]}
cv = KFold(n_splits=5, shuffle=True, random_state=seed)
gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='r2')

gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)

y_test_pred = gs.predict(X_test)

r2_test = r2_score(y_true=y_test, y_pred=y_test_pred)
print(f"r2score...{r2_test}")
