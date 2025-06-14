import numpy as np
def sigmoid_fn(t, a, b, c):
    return a / (1 + np.exp(-b * (t - c)))

def linear_fn(t, a, b):
    return a * t + b

def exp_fn(t, a, b):
    return a * np.exp(b * t)

def dec_exp_fn(t, a, b, c, d):
    f = a * (1 - np.exp(-b * t + b * c))
    f[f < d] = d
    return f

def log_fn(t, a, b):
    return a * np.log(t) + b