from data_generation import dec_exp_fn, linear_fn
from preprocessing import preprocessor
import numpy as np
import pandas as pd
import torch

def _weights(x, tau, mu):
    raw = np.array([np.exp(-((x - m) ** 2) / tau**2) for m in mu])
    return raw / np.sum(raw, axis=0)

def piecewise_yield_time_x(x_scalar, t, params, mu):
    x_scalar = np.asarray(x_scalar)  # shape (n_obs,)
    mu = np.sort(np.asarray(mu))     # ensure mu is sorted
    breakpoints = (mu[:-1] + mu[1:]) / 2  # midpoints between mu_i

    # Append -inf and +inf to make interval comparison easier
    all_bounds = np.concatenate(([-np.inf], breakpoints, [np.inf]))  # shape (n+1,)

    n_obs = x_scalar.shape[0]
    yield_val = np.zeros((n_obs, len(t)))

    all_params = params['linear'] + params['dec exp']

    for i in range(len(all_params)):
        # Determine which x fall into the i-th region
        in_region = (x_scalar >= all_bounds[i]) & (x_scalar < all_bounds[i+1])

        if i < len(params['linear']):
            a, b = params['linear'][i]
            g_i = linear_fn(t, a, b)  # shape (len(t),)
        else:
            a, b, c, d = params['dec exp'][i - len(params['linear'])]
            g_i = dec_exp_fn(t, a, b, c, d)  # shape (len(t),)

        # Broadcast g_i to each selected observation
        yield_val[in_region, :] = g_i

    return yield_val

class RoelfesEmulator:

    def __init__(self, dim=1, objective="normalized yield gradient", discrete_inds=None, t_max=5000, 
                 time_points=100, space_complexity = 4, group_count=5, use_torch=False):
        """
        objective : is  a string from ["coef", "max yield", "normalized yield gradient"] 
        if coef, we return a multi objective problem where dec_exp_c must be minimized and 
        the rest must be maximized.
        """
        if discrete_inds is None:
            discrete_inds=[]
        self.use_torch = use_torch
        self.dim = dim
        self.t_vals = np.linspace(0.01, t_max, time_points)
        self.t_max = t_max
        self.discrete_inds = discrete_inds,
        self.space_complexity = space_complexity
        self.get_random_params()
        self.objective = objective
        self.group_count = group_count
        lower = torch.full((dim,), -100)
        upper = torch.full((dim,), 100)
        self.bounds = torch.stack([lower, upper])
        for idx in self.discrete_inds:
            self.bounds[0, idx] = 0.0
            self.bounds[1, idx] = 1.0 
        self.obj_dict = {"coef" : ['linear_a','linear_b','dec_exp_a','dec_exp_b','dec_exp_c','dec_exp_d'],
                        "max yield" : ['max_yield'],
                        "normalized yield gradient" : ['norm_yield_grad'],}

    def generate_random_params(self, space_complexity):
        # Random split of indices
        indices = np.random.permutation(space_complexity)
        n_linear = np.random.randint(1, space_complexity)  # ensure at least 1 of each type
        linear_idx = indices[:n_linear]
        dec_exp_idx = indices[n_linear:]

        linear_params = []
        for _ in linear_idx:
            b = np.random.uniform(0, 20)
            a = np.random.uniform(0, (100 - b) / self.t_max)
            linear_params.append((a, b))
        dec_exp_params = [(np.random.uniform(1e-10,100), np.random.uniform(1e-10,0.05),
                           np.random.uniform(1e-10, 2 * self.t_max / 3), np.random.uniform(1e-10, 20))
                           for _ in dec_exp_idx]

        return {'linear': linear_params, 'dec exp': dec_exp_params}

    def get_random_params(self):
        self.mu = sorted(np.random.uniform(0, 100, self.space_complexity).tolist())
        self.random_weights = np.random.exponential(scale=1.0, size=self.dim)
        self.random_weights /= np.sum(self.random_weights)
        self.params = self.generate_random_params(self.space_complexity)
        

    def yield_calc(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        X_min, X_max = x.min(), x.max()
        if X_max > X_min:
            X = (x - X_min) / (X_max - X_min)
            X = X * 100
        else:
            X = np.zeros_like(X)

        X = np.dot(X, self.random_weights)

        yield_val = piecewise_yield_time_x(X, self.t_vals, self.params, self.mu)

        yield_val[yield_val < 0] = 0

        return yield_val
    
    def prepare_df_cols(self, n):
        wells_per_group = int(np.ceil(n / self.group_count))

        well_names = []

        for i in range(self.group_count):
            label = chr(65 + i)  # A, B, C...
            for j in range(wells_per_group):
                idx = i * wells_per_group + j
                if idx >= n:
                    break
                well_names.append(f"{label}{j+1}")
        return well_names

    def __call__(self, x, plot=False):

        if self.use_torch is True:
            x.numpy()
        yields = self.yield_calc(x)
        numpy_pre_df = np.vstack((self.t_vals, np.zeros_like(self.t_vals),  yields))
        n = x.shape[0]
        well_names = self.prepare_df_cols(n)

        raw_cols = ['Time'] + ['T_Read'] + well_names
        var_names = [f'x{i+1}' for i in range(x.shape[1])]

        raw_em_df = pd.DataFrame(numpy_pre_df.T, columns=raw_cols)
        settings_cols = ['well'] + well_names
        settings_em_df = pd.DataFrame(np.vstack((var_names, x)).T, columns=settings_cols)
        settings_em_df.iloc[:,1:] = settings_em_df.iloc[:,1:].astype(float)
        objectives = preprocessor(settings_em_df, raw_em_df, pos_wells=well_names, plot=plot, 
                     return_coef=True, return_function_type=False)
        relevant_objectives = objectives[self.obj_dict[self.objective]]
        if self.use_torch is True:
            relevant_objectives = torch.tensor(relevant_objectives.values)

        return relevant_objectives
        