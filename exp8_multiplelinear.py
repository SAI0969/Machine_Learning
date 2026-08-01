
import numpy as np

# Dataset
X = np.array([
    [1, 2],
    [2, 4],
    [3, 5]
], dtype=float)

Y = np.array([5, 7, 9], dtype=float)

# Initialize parameters
w1 = 0
w2 = 0
b = 0

learning_rate = 0.01
epochs = 3
n = len(X)

# Gradient Descent
for epoch in range(epochs):

    # Predicted values
    Y_pred = w1 * X[:, 0] + w2 * X[:, 1] + b

    # Error
    error = Y_pred - Y

    # Cost (Mean Squared Error)
    loss = np.mean(error ** 2)

    # Gradients
    dw1 = (2 / n) * np.sum(error * X[:, 0])
    dw2 = (2 / n) * np.sum(error * X[:, 1])
    db = (2 / n) * np.sum(error)

    # Update parameters
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b = b - learning_rate * db


# Final parameters
print("\nFinal Parameters")
print("Weight1 =", w1)
print("Weight2 =", w2)
print("Bias =", b)

# Prediction
x1 = 5
x2 = 6

prediction = w1 * x1 + w2 * x2 + b

print("\nPrediction for x1 =", x1, "x2 =", x2)
print("Predicted Value =", prediction)
