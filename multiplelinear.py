import numpy as np

X = np.array([
    [2, 7],
    [3, 6],
    [4, 8],
    [5, 7],
    [6, 9]
], dtype=float)


Y = np.array([65, 70, 80, 85, 95], dtype=float)


weight1 = 0
weight2 = 0
bias = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

for epoch in range(epochs):

   
    Y_pred = (weight1 * X[:,0]) + (weight2 * X[:,1]) + bias

    error = Y_pred - Y

    
    dw1 = (2 / n) * np.sum(error * X[:,0])
    dw2 = (2 / n) * np.sum(error * X[:,1])
    db = (2 / n) * np.sum(error)

   
    weight1 = weight1 - learning_rate * dw1
    weight2 = weight2 - learning_rate * dw2
    bias = bias - learning_rate * db

print("Weight1 :", weight1)
print("Weight2 :", weight2)
print("Bias :", bias)

new_x1 = 5     
new_x2 = 8     

prediction = (weight1 * new_x1) + (weight2 * new_x2) + bias

print("The predicted value is", prediction)
