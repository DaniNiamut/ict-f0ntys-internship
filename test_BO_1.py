import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Generate some example data (log-spaced for parameter tuning scenarios)
X = np.logspace(-3, 0, 5).reshape(-1, 1)  # X values spread logarithmically
y = np.sin(2 * np.pi * X).ravel()  # Some nonlinear function

# Define Gaussian Process kernel (RBF)
kernel = C(1.0) * RBF(length_scale=0.1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

# Train the Gaussian Process model
gp.fit(X, y)

# Create test points for predictions
X_pred = np.logspace(-3, 0, 100).reshape(-1, 1)

# Predict mean and uncertainty
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Plot results
plt.figure(figsize=(8, 4))
plt.xscale("log")  # Set x-axis to log scale
plt.plot(X_pred, y_pred, 'k-', label="Mean Prediction")  # Solid black line for the mean
plt.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                 alpha=0.2, color='blue', label="95% Confidence Interval")  # Shaded region
plt.scatter(X, y, color='black', zorder=3)  # Observed data points
plt.xlabel("Parameter")
plt.legend()
plt.show()
