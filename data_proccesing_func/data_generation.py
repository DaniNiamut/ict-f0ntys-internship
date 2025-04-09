import numpy as np

def sigmoid_fn(t, a, b, c):
    return a / (1 + np.exp(-b * (t - c)))

def linear_fn(t, a, b):
    return a * t + b

def exp_fn(t, a, b):
    return a * np.exp(b * t)

def log_fn(t, a, b):
    return a * np.log(t) + b

def weights(x, tau, mu):
    raw = np.array([np.exp(-((x - m) ** 2) / tau**2) for m in mu])
    return raw / np.sum(raw, axis=0)

def yield_time_x(x, t, params, tau, mu):
    w = weights(x, tau, mu)
    a1, b1, c1 = params['sigmoid']
    a2, b2 = params['linear']
    a3, b3 = params['exp']
    a4, b4 = params['log']
    
    return (w[0] * sigmoid_fn(t, a1, b1, c1) +
            w[1] * linear_fn(t, a2, b2) +
            w[2] * exp_fn(t, a3, b3) +
            w[3] * log_fn(t, a4, b4))