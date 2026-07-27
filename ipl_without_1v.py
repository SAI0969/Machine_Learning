import pandas as pd
import numpy as np

df = pd.read_csv("ipl2013.csv")

numeric = df.select_dtypes(include=np.number)


X = numeric.drop(columns=["SOLD PRICE"]).values
y = numeric["SOLD PRICE"].values

weights = np.zeros(X.shape[1])
bias = 0

learning_rate = 0.01
epochs = 4

n = len(X)

for epoch in range(epochs):

    y_pred = np.dot(X, weights) + bias

    error = y_pred - y

    dw = (2 / n) * np.dot(X.T, error)
    db = (2 / n) * np.sum(error)


    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    print("Iteration:", epoch + 1)
    print("Weights:", weights)
    print("Bias:", bias)
    print()

print("Final Weights:", weights)
print("Final Bias:", bias)
