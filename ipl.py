import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv("ipl2013.csv")

# Fill missing numeric values
df = df.fillna(df.mean(numeric_only=True))

# Keep only numeric columns
numeric = df.select_dtypes(include=np.number)

# Correlation Matrix (Optional)
corr = numeric.corr()
print(corr)

# Features and Target
X = numeric.drop(columns=["SOLD PRICE"]).values
y = numeric["SOLD PRICE"].values

# Add bias column (Intercept)
X = np.c_[np.ones(len(X)), X]

# Train-Test Split (80:20)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

# Normal Equation
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Prediction
prediction = X_test @ beta

# RMSE
rmse = np.sqrt(np.mean((y_test - prediction) ** 2))

print("\nRegression Coefficients:")
print(beta)

print("\nRMSE:", rmse)
