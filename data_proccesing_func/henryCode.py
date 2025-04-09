import logging
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.preprocessing import MinMaxScaler

# Surrogate model for predictions
def surrogate(model, X):
    """Predict the mean and standard deviation from the GP model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    return model.predict(X, return_std=True)

# Surrogate model for predictions
def fit_model(model, X, y):
    """fit the model"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    return model.fit(X, y)

# Probability of Improvement (PI) acquisition function
def acquisition_PI(X, Xsamples, model):
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    xi = 0.01
    return norm.cdf((mu - best - xi) / (std + 1E-9))

# Expected Improvement (EI) acquisition function
def acquisition_EI(X, Xsamples, model):
    yhat, _ = surrogate(model, X)
    best = max(yhat)
    mu, std = surrogate(model, Xsamples)
    xi = 0
    Z = (mu - best - xi) / (std + 1E-9)
    return (mu - best - xi) * norm.cdf(Z) + std * norm.pdf(Z)

# Upper Confidence Bound (UCB) acquisition function
def acquisition_UCB(X, Xsamples, model):
    mu, std = surrogate(model, Xsamples)
    kappa = 2
    return mu + kappa * std

# Optimize acquisition function using Scipy's 'minimize' method with multiple restarts
def optimize_acquisition(X, model, dims, acquisition="EI", num_restarts=10):
    """Optimize the acquisition function using multiple starting points to avoid local optima."""
    
    def acquisition_function(Xsample):
        """Negative acquisition function (since we minimize)."""
        Xsample = Xsample.reshape(1, -1)
        if acquisition == "PI":
            return -acquisition_PI(X, Xsample, model)
        elif acquisition == "EI":
            return -acquisition_EI(X, Xsample, model)
        elif acquisition == "UCB":
            return -acquisition_UCB(X, Xsample, model)

    # Define parameter bounds
    bounds = list(zip(dims.min().values, dims.max().values))
    
    # Generate multiple random starting points
    initial_guesses = np.random.uniform(dims.min().values, dims.max().values, size=(num_restarts, len(dims.columns)))
    
    best_X = None
    best_acq_value = float("inf")  # Since we are minimizing, start with a large value

    for guess in initial_guesses:
        res = minimize(acquisition_function, guess, method="L-BFGS-B", bounds=bounds)

        # Keep the best result found
        if res.fun < best_acq_value:
            best_acq_value = res.fun
            best_X = res.x

    return best_X


# Bayesian Optimization Function
def bayesian_optimization(dims, df_obs, batchsize=5, target='rate', acquisition='EI', 
                          length_scale_Matern=1.0, noise_level_bounds=(0., 0.), scale_obs=False):
    """Performs Bayesian Optimization using a Gaussian Process with Scipy-based acquisition function optimization."""
    
    # Setup logging
    logging.basicConfig(filename="warnings.log", level=logging.WARNING, format="%(asctime)s - %(message)s")

    # Function to log warnings instead of printing them
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"WARNING: {message} (Category: {category._name_}, File: {filename}, Line: {lineno})")
        
    # Check if target variable is correctly specified
    if target not in df_obs.columns:
        raise Exception('Invalid target variable specified!')
    
    parameters = dims.columns.to_list()
    X = df_obs[parameters].to_numpy()
    y = df_obs[[target]].to_numpy()
    
    dX = dims[parameters].loc[2].to_numpy()
    
    # Normalize observations
    if scale_obs:
        y = MinMaxScaler().fit_transform(y)

    # Apply custom warning handler
    warnings.showwarning = warning_handler
    with warnings.catch_warnings():
        warnings.simplefilter("always")  # Ensure warnings are always caught
        
        # Define Gaussian Process Model with Matern 5/2 Kernel
        if min(noise_level_bounds) < 1e-12:
            gp_kernel = Matern(length_scale=length_scale_Matern, nu=2.5, length_scale_bounds=(1e-10, 1e5))
            model = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-10)  # Small alpha for numerical stability
        else:
            noise_level_WK = np.exp(np.log(noise_level_bounds).mean())
            gp_kernel = Matern(length_scale=length_scale_Matern, nu=2.5, length_scale_bounds=(1e-10, 1e5)) \
                         + WhiteKernel(noise_level=noise_level_WK, noise_level_bounds=noise_level_bounds)
            model = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-10)  # Small alpha for numerical stability
    
    Xbatch = X.copy()

    # Bayesian Optimization Batching Loop
    i = 0
    j = 0
    while i < batchsize:
        fit_model(model, Xbatch, y)
        
        # Optimize acquisition function to get next X
        next_X = optimize_acquisition(Xbatch, model, dims, acquisition)
        
        # Project onto specified parameter grid
        next_X_on_grid = np.round(next_X/dX)*dX
        
        j += 1
        print(j)
        # Ensure new points are unique
        if not np.any(np.all(next_X_on_grid == Xbatch, axis=1)):
            Xbatch = np.vstack((Xbatch, next_X_on_grid))
            y_pred = model.predict(next_X_on_grid.reshape(1, -1))
            y = np.vstack((y, y_pred))
            i += 1

    # Prepare the final output
    df_proposed = pd.DataFrame(Xbatch[-batchsize:, :], columns=parameters)
    
    return df_proposed


