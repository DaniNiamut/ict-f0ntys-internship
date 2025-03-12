import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Generate sample dataX = np.linspace(-5, 5, 10).reshape(-1, 1)
X = np.linspace(-5, 5, 10).reshape(-1, 1)

y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Noisy observations

# Define the Gaussian Process model with an RBF kernel
kernel = C(1) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

# Fit to the data
gp.fit(X, y)

# Make predictions
X_pred = np.linspace(-6, 6, 100).reshape(-1, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, c='r', label="Observations")
plt.plot(X_pred, np.sin(X_pred), 'g--', label="True function")
plt.plot(X_pred, y_pred, 'b', label="GP Mean prediction")
plt.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, 
                 alpha=0.2, color='blue', label="95% Confidence Interval")
plt.legend()
plt.title("Gaussian Process Regression")



plt.show()
