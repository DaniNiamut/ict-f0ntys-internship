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
from data_generation import sigmoid_fn, linear_fn, exp_fn, log_fn, yield_time_x, neg_exp_fn

def least_squares_fitter(t_vals,y_vals):
    params = {}
    scores = []

    # Fitting sigmoidal function. (Manually calculate L, and t0)
    L = max(y_vals)
    slopes  = np.diff(y_vals) / np.diff(t_vals)
    t0 = max(slopes)
    z = np.log(L / y_vals - 1 + 1e10)
    X = (t_vals - t0).reshape(-1,1)

    model = LinearRegression(fit_intercept=False).fit(X, z)
    coef, const, score = model.coef_[0], model.intercept_, model.score(X, z)
    params['sigmoid'] = L, coef, t0
    scores.append(score)

    #Fitting linear model
    X = t_vals.reshape(-1,1)

    model = LinearRegression().fit(X, y_vals)
    coef, const, score = model.coef_[0], model.intercept_, model.score(X, y_vals)
    params['linear'] = coef, const
    scores.append(score)

    #Fitting exponential model
    z = np.log(y_vals)
    X = t_vals.reshape(-1, 1)

    model = LinearRegression().fit(X, z)
    coef, const, score = model.coef_[0], np.exp(model.intercept_), model.score(X, z)
    params['exp'] = coef, const
    scores.append(score)

    #Fitting negative exponential model
    a = max(y_vals)
    d = min(y_vals)
    slopes = np.diff(y_vals) / np.diff(t_vals)
    x0_ind = np.argmax(slopes) - 2
    c = t_vals[x0_ind]

    z = np.log(a - y_vals[x0_ind:] + 1e-5)
    X = (t_vals[x0_ind:] - t_vals[x0_ind]).reshape(-1, 1)
    model = LinearRegression().fit(X, z)

    b = -model.coef_[0]

    y_pred = neg_exp_fn(t_vals,a ,b ,c ,d)
    score = r2_score(y_vals, y_pred)
    params['neg exp'] = (a, b, c, d)
    scores.append(score)

    return params, scores

def preprocessor(settings_df, raw_df, pos_wells, override_wells=None, plot=False):
    params_names = ['sigmoid_a', 'sigmoid_b', 'sigmoid_c', 'linear_a', 'linear_b', 'exp_a',
               'exp_b', 'neg_exp_a', 'neg_exp_b', 'neg_exp_c', 'neg_exp_d']
    params_indices = {'sigmoid': [0, 1, 2],
        'linear': [3, 4],
        'exp': [5, -5],
        'neg exp': [-4, -3, -2, -1]}
    params_functs = {'sigmoid': sigmoid_fn,
        'linear': linear_fn,
        'exp': exp_fn,
        'neg exp': neg_exp_fn}

    cols = list(raw_df.columns)
    wells = cols[2:]
    max_react_rates = []
    yields = []
    best_fits = []
    params = {}
    all_params = []
    all_params_lite = []

    settings_df = settings_df.set_index('well').T.reset_index()
    settings_df.columns.name = None
    settings_df = settings_df.drop(columns='index')
    time_column = raw_df['Time'].apply(lambda t: (t.hour * 3600 + t.minute * 60 + t.second) / 86400).to_numpy()

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

    #for konstantin
    if plot:
        time_column # use this as time
        y_fit = np.array([params_functs[best_fits[i]](time_column, *all_params_lite[i])
                           for i in range(len(best_fits))]).T
        y_true = raw_df[wells].to_numpy()
        norm_react_rates # use this as red numbers
        #norm_react_rates and time_column are 1d numpy arrays.
        #y_fit and y_true are numpy arrays of cols representing wells and rows being time.

        #plot(time_column,y_true, y_fit, norm_react_rates)
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

    def visualize(black_box_f, projection='2D', return_frame=False):
        '''
        Create 2D or 3D plot using reduced dimensions of the acquisition function 
        and surrogate and black_box_f if provided.


        '''






