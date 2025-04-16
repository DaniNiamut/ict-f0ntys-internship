import torch
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression
from data_generation import sigmoid_fn, linear_fn, exp_fn, log_fn, yield_time_x, dec_exp_fn
import string
from scipy.optimize import curve_fit

def plot_well_data(time_column, y_true, y_filtered, y_fit, norm_react_rates, wells):
    max_row = max(w[0] for w in wells)
    max_col = max(int(w[1:]) for w in wells)

    rows = string.ascii_uppercase.index(max_row) + 1
    cols = max_col

    well_to_index = {w: i for i, w in enumerate(wells)}

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=False)
    axes = np.array(axes).reshape((rows, cols))

    for i in range(rows):
        for j in range(cols):
            row_letter = string.ascii_uppercase[i]
            col_number = j + 1
            well_name = f"{row_letter}{col_number}"
            ax = axes[i, j]

            if well_name in well_to_index:
                idx = well_to_index[well_name]
                ax.plot(time_column, y_true[:, idx], label='Raw', color='#FF7F0E')      # orange
                ax.plot(time_column, y_filtered[:, idx], '--',  label='Filtered', color='#4CAF50', alpha=0.5)  # green
                ax.plot(time_column, y_fit[:, idx], '-', label='Fit', color='#257BB6', alpha=0.5)       # blue

                ax.text(0.1, 0.9, well_name, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='top', ha='left')
                ax.text(0.1, 0.70, f"{norm_react_rates[idx]:.3f}", transform=ax.transAxes,
                        fontsize=10, va='top', ha='left', color='red')
            else:
                ax.set_visible(False)

            ax.tick_params(labelsize=6)

    fig.suptitle("Yield over Time for all wells")
    fig.supylabel("Yield")
    fig.supxlabel("Time (s)")

    fig.text(0.78, 0.975, "Legend:", color='black')
    fig.text(0.84, 0.98, "Raw", fontweight='bold', color='#FF7F0E')
    fig.text(0.84, 0.96, "Filtered", fontweight='bold', color='#4CAF50')
    fig.text(0.84, 0.94, "Fit", fontweight='bold', color='#257BB6')
    fig.text(0.91, 0.97, "Reaction Rate", fontweight='bold', color='red')

    plt.tight_layout()
    plt.show()

def least_squares_fitter(t_vals, y_vals):
    t_vals = np.asarray(t_vals).reshape(-1)
    y_vals = np.asarray(y_vals).reshape(-1)

    params = {}
    scores = []

    # Sigmoid approximation
    L = np.max(y_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        y_ratio = np.clip(L / y_vals - 1, 1e-10, None)
        z = np.log(y_ratio)
    # Estimate t0 as the t at max slope
    slopes = np.gradient(y_vals, t_vals)
    t0_index = np.argmax(slopes)
    t0 = t_vals[t0_index]
    X = (t_vals - t0).reshape(-1, 1)

    model = LinearRegression(fit_intercept=False).fit(X, z)
    k = model.coef_[0]
    params['sigmoid'] = (L, k, t0)
    y_pred = sigmoid_fn(t_vals, *params['sigmoid'])
    sigmoid_score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    scores.append(sigmoid_score)

    # Linear
    X = t_vals.reshape(-1, 1)
    model = LinearRegression().fit(X, y_vals)
    coef, intercept = model.coef_[0], model.intercept_
    params['linear'] = (coef, intercept)
    y_pred = linear_fn(t_vals, *params['linear'])
    linear_score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    scores.append(linear_score)

    # Exponential
    with np.errstate(divide='ignore'):
        z = np.log(np.clip(y_vals, 1e-10, None))
    model = LinearRegression().fit(X, z)
    b = np.exp(model.intercept_)
    a = model.coef_[0]
    params['exp'] = (a, b)  # y = a * exp(b*t)
    y_pred = exp_fn(t_vals, *params['exp'])
    exp_score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    scores.append(exp_score)

    # Negative Exponential
    a = np.max(y_vals)
    d = np.min(y_vals)
    slopes = np.gradient(y_vals, t_vals)
    peak_index = np.argmax(slopes)
    c = t_vals[peak_index]

    y_tail = y_vals[peak_index:]
    t_tail = t_vals[peak_index:]
    with np.errstate(divide='ignore'):
        z = np.log(np.clip(a - y_tail, 1e-10, None))
    X_tail = (t_tail - c).reshape(-1, 1)

    model = LinearRegression().fit(X_tail, z)
    b = -model.coef_[0]
    params['dec exp'] = (a, b, c, d)
    y_pred = dec_exp_fn(t_vals, *params['dec exp'])
    dec_exp_score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    scores.append(dec_exp_score)

    return params, scores

def least_squares_dec_exp(t_vals, y_vals):
    t_vals = np.asarray(t_vals).reshape(-1)
    y_vals = np.asarray(y_vals).reshape(-1)

    a = np.max(y_vals)
    d = np.min(y_vals)
    slopes = np.gradient(y_vals, t_vals)
    peak_index = np.argmax(slopes)
    c = t_vals[peak_index]

    y_tail = y_vals[peak_index:]
    t_tail = t_vals[peak_index:]
    with np.errstate(divide='ignore'):
        z = np.log(np.clip(a - y_tail, 1e-10, None))
    X_tail = (t_tail - c).reshape(-1, 1)

    model = LinearRegression().fit(X_tail, z)
    b = -model.coef_[0]
    params = (a, b, c, d)
    y_pred = dec_exp_fn(t_vals, *params)
    score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    return params, score

def preprocessor(settings_df, raw_df, pos_wells, override_wells=None, plot=False, ):
    cols = list(raw_df.columns)
    wells = cols[2:]
    max_react_rates = []
    yields = []
    all_params = []
    x_fits = []

    settings_df = settings_df.set_index('well').T.reset_index()
    settings_df.columns.name = None
    settings_df = settings_df.drop(columns='index')
    time_column = raw_df['Time'].apply(lambda t: (t.hour * 3600 + t.minute * 60 + t.second)).to_numpy()

    for well in wells:
        y_vals = raw_df[well].to_numpy()

        ir = IsotonicRegression(increasing=True)
        x_fit = ir.fit_transform(time_column, y_vals)
        total_yield = max(y_vals)
        slopes = np.diff(x_fit) / np.diff(time_column)
        t0 = max(slopes)

        params, _ = least_squares_dec_exp(time_column, x_fit)

        max_react_rates.append(t0)
        yields.append(total_yield)
        all_params.append(params)
        x_fits.append(x_fit)

    max_react_rates = np.array(max_react_rates)
    if pos_wells:
        pos_well_ind = [i for i, col in enumerate(wells) if col in pos_wells]
        pos_wells_t0 = max_react_rates[pos_well_ind].mean()
        norm_react_rates = max_react_rates / pos_wells_t0
    else:
        norm_react_rates = max_react_rates

    settings_df[['dec_exp_a', 'dec_exp_b', 'dec_exp_c', 'dec_exp_d']] = np.array(all_params)
    settings_df['norm_yield_grad'] = norm_react_rates
    settings_df['max_yield'] = yields

    if plot:
        y_fit = np.array([dec_exp_fn(time_column, *all_params[i])
                           for i in range(len(all_params))]).T
        y_true = raw_df[wells].to_numpy()
        y_filtered = np.array(x_fits).T
        plot_well_data(time_column, y_true, y_filtered, y_fit, norm_react_rates, wells)
    return settings_df

def preprocessor_any_func(settings_df, raw_df, pos_wells, override_wells=None, plot=False, ):
    params_names = ['sigmoid_a', 'sigmoid_b', 'sigmoid_c', 'linear_a', 'linear_b', 'exp_a',
               'exp_b', 'dec_exp_a', 'dec_exp_b', 'dec_exp_c', 'dec_exp_d']
    params_indices = {'sigmoid': [0, 1, 2],
        'linear': [3, 4],
        'exp': [5, -5],
        'dec exp': [-4, -3, -2, -1]}
    params_functs = {'sigmoid': sigmoid_fn,
        'linear': linear_fn,
        'exp': exp_fn,
        'dec exp': dec_exp_fn}

    cols = list(raw_df.columns)
    wells = cols[2:]
    max_react_rates = []
    yields = []
    best_fits = []
    params = {}
    all_params = []
    all_params_lite = []
    x_fits = []

    settings_df = settings_df.set_index('well').T.reset_index()
    settings_df.columns.name = None
    settings_df = settings_df.drop(columns='index')
    time_column = raw_df['Time'].apply(lambda t: (t.hour * 3600 + t.minute * 60 + t.second)).to_numpy()

    for well in wells:
        params_as_vars = np.zeros(11)
        y_vals = raw_df[well].to_numpy()

        ir = IsotonicRegression(increasing=True)
        x_fit = ir.fit_transform(time_column, y_vals)
        total_yield = max(y_vals)
        slopes = np.diff(x_fit) / np.diff(time_column)
        t0 = max(slopes)

        params, scores = least_squares_fitter(time_column, x_fit)
        best_fit = list(params.keys())[np.argmax(scores)]
        params_as_vars[params_indices[best_fit]] = params[best_fit]

        max_react_rates.append(t0)
        yields.append(total_yield)
        best_fits.append(best_fit)
        all_params.append(params_as_vars)
        all_params_lite.append(params[best_fit])
        x_fits.append(x_fit)

    max_react_rates = np.array(max_react_rates)
    if pos_wells:
        pos_well_ind = [i for i, col in enumerate(wells) if col in pos_wells]
        pos_wells_t0 = max_react_rates[pos_well_ind].mean()
        norm_react_rates = max_react_rates / pos_wells_t0
    else:
        norm_react_rates = max_react_rates

    settings_df[params_names] = all_params
    settings_df['norm_yield_grad'] = norm_react_rates
    settings_df['max_yield'] = yields
    settings_df['function_type'] = best_fits

    if plot:
        y_fit = np.array([params_functs[best_fits[i]](time_column, *all_params_lite[i])
                           for i in range(len(best_fits))]).T
        y_true = raw_df[wells].to_numpy()
        y_filtered = np.array(x_fits).T
        plot_well_data(time_column, y_true, y_filtered, y_fit, norm_react_rates, wells)
    return settings_df

class MCBayesianOptimization:
    """
    Monte-Carlo based single-objective bayesian optimization.

    MCBayesianOptimization fits a Gaussian process regressor to suggest
    candidate points for a following iteration. It does this as a middle-man
    between NumPy and PyTorch while attempting to resemble sklearn regressors
    in terms of user interaction.

    Attributes
    ----------
    gp_mean
    gp_cov
    gp

    """

    def __init__(self):
        self.gp_mean = None
        self.gp_cov = None
        self.gp = None
        self.ucb_hyperparam = 0.1
        self.num_restarts = 5
        self.raw_samples = 20

    def fit(self, X, y, cat_dims=None, kernel=None):
        """
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features) of training data.

        y : list of strings of the names corresponding to the columns for target data found in X.

        cat_dims : list of names corresponding to the columns of the input X that should be considered categorical features.
        
        """
        train_x = X.drop(y, axis=1)
        self.var_names = list(train_x.columns)
        train_y = X[y]
        train_x = train_x.to_numpy().reshape(-1,np.shape(train_x)[1])
        train_y = train_y.to_numpy().reshape(-1,np.shape(train_y)[1])
        self.train_x = torch.tensor(train_x)
        self.train_y = torch.tensor(train_y)
        self.y_names = y

        if cat_dims:
            cat_dims = [self.var_names.index(v) for v in cat_dims]
            self.gp = MixedSingleTaskGP(self.train_x, self.train_y, cat_dims=cat_dims, cont_kernel_factory=kernel)
        else:
            self.gp = SingleTaskGP(
                train_X=self.train_x,
                train_Y=self.train_y,
                input_transform=Normalize(d=np.shape(self.train_x)[1]),
                outcome_transform=Standardize(m=1),
                covar_module=kernel
            )

        return self
    
    def candidates(self, q, acq_func_name='LogEI', bounds=None, export_df=False):
        """
        Optimizes an acquisition function to return candidates
        """
        acq_dict = {
        'LogEI': [LogExpectedImprovement, self.train_y.max()],
        'UCB': [UpperConfidenceBound, 0.1],
        'LogPI': [ProbabilityOfImprovement, self.train_y.max()],
        }
        q_acq_dict = {
        'LogEI': [qLogExpectedImprovement, self.train_y.max()],
        'UCB': [qUpperConfidenceBound, 0.1],
        'LogPI': [qProbabilityOfImprovement, self.train_y.max()],
        }

        if q > 1:
            self.acq_func = q_acq_dict[acq_func_name][0](self.gp, acq_dict[acq_func_name][1])
        else:
            self.acq_func = acq_dict[acq_func_name][0](self.gp, acq_dict[acq_func_name][1])

        if bounds:
            self.bounds = torch.tensor(bounds)
        else:
            bounds = (torch.min(self.train_x, 0)[0], torch.max(self.train_x, 0)[0])
            self.bounds = torch.stack(bounds)

        candidate, _ = optimize_acqf(self.acq_func, bounds=self.bounds, q=q, 
                                             num_restarts=self.num_restarts, raw_samples=self.raw_samples)
        
        prediction = self.gp.posterior(candidate).mean

        candidate, prediction = candidate.detach().numpy(), prediction.detach().numpy()
        
        if export_df:
            pred_df = pd.DataFrame(prediction, columns=self.y_names)
            candidate_df = pd.DataFrame(candidate, columns=self.var_names)
            suggested_df = pd.concat((candidate_df, pred_df),axis=1)
            return suggested_df
        else:
            return candidate, prediction

    #def visualize(black_box_f=None, projection='2D', return_frame=False):






