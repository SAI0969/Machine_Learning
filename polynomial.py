print("Polynomial")

import numpy as np

X = np.array([
    [2, 7],
    [3, 6],
    [4, 8],
    [5, 7],
    [6, 9]
], dtype=float)

Y = np.array([65, 70, 80, 85, 95], dtype=float)

x1 = X[:, 0]
x2 = X[:, 1]
x1_sq = x1 ** 2
x2_sq = x2 ** 2

weight1 = 0
weight2 = 0
weight3 = 0
weight4 = 0
bias = 0

learning_rate = 0.0001
epochs = 1000
n = len(X)

for epoch in range(epochs):

    Y_pred = (weight1 * x1) + (weight2 * x2) + \
             (weight3 * x1_sq) + (weight4 * x2_sq) + bias

    error = Y_pred - Y

    dw1 = (2 / n) * np.sum(error * x1)
    dw2 = (2 / n) * np.sum(error * x2)
    dw3 = (2 / n) * np.sum(error * x1_sq)
    dw4 = (2 / n) * np.sum(error * x2_sq)
    db = (2 / n) * np.sum(error)

    weight1 = weight1 - learning_rate * dw1
    weight2 = weight2 - learning_rate * dw2
    weight3 = weight3 - learning_rate * dw3
    weight4 = weight4 - learning_rate * dw4
    bias = bias - learning_rate * db

print("Weight1 (x1):", weight1)
print("Weight2 (x2):", weight2)
print("Weight3 (x1²):", weight3)
print("Weight4 (x2²):", weight4)
print("Bias:", bias)

new_x1 = 5
new_x2 = 8

prediction = (weight1 * new_x1) + \
             (weight2 * new_x2) + \
             (weight3 * new_x1**2) + \
             (weight4 * new_x2**2) + bias


print("The predicted value is:", prediction)

rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
print("RMSE:", rmse)
