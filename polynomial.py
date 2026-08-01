import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([2, 5, 10, 17, 26], dtype=float)

# Initialize parameters
w1 = 0
w2 = 0
b = 0

learning_rate = 0.01
epochs = 4
n = len(X)

# Gradient Descent
for epoch in range(epochs):

    # Prediction
    Y_pred = w1 * X + w2 * (X ** 2) + b

    # Error
    error = Y_pred - Y

    # Cost
    loss = np.mean(error ** 2)

    # Gradients
    dw1 = (2 / n) * np.sum(error * X)
    dw2 = (2 / n) * np.sum(error * (X ** 2))
    db = (2 / n) * np.sum(error)

    # Update parameters
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db

    if epoch % 1 == 0:
        print(f"Epoch {epoch}  Loss = {loss:.4f}")

print("\nFinal Parameters")
print("Weight for x =", w1)
print("Weight for x² =", w2)
print("Bias =", b)

# Prediction
new_x = 6
prediction = w1 * new_x + w2 * (new_x ** 2) + b

print("\nPrediction for x =", new_x)
print("Predicted Value =", prediction)
