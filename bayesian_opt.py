import torch
from botorch.models import SingleTaskGP, MixedSingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP

from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement

from botorch.optim import optimize_acqf
import numpy as np
import matplotlib.pyplot as plt

from botorch import fit_fully_bayesian_model_nuts
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression
from data_generation import sigmoid_fn, linear_fn, exp_fn, dec_exp_fn
import string

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

def general_saasbo_gp(X, Y):
    models_list = []
    for i in range(Y.shape[1]):
        model = SaasFullyBayesianSingleTaskGP(X, Y[:,i].reshape(-1,1))
        fit_fully_bayesian_model_nuts(model, disable_progbar=True)
        models_list.append(model)
    if len(models_list)> 1:
        models_list = tuple(models_list)
        gp = ModelListGP(*models_list)
    else:
        gp = models_list[0]
    return gp


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

def preprocessor(settings_df, raw_df, pos_wells, override_wells=None, plot=False, return_coef=False):
    # Only keep linear and decaying exponential parameters
    params_names = ['linear_a', 'linear_b', 'dec_exp_a', 'dec_exp_b', 'dec_exp_c', 'dec_exp_d']
    params_indices = {
        'linear': [0, 1],
        'dec exp': [2, 3, 4, 5]
    }
    params_functs = {
        'linear': linear_fn,
        'dec exp': dec_exp_fn
    }

    cols = list(raw_df.columns)
    wells = cols[2:]
    max_react_rates = []
    yields = []
    best_fits = []
    all_params = []
    all_params_lite = []
    x_fits = []

    settings_df = settings_df.set_index('well').T.reset_index()
    settings_df.columns.name = None
    settings_df = settings_df.drop(columns='index')
    time_column = raw_df['Time'].apply(lambda t: (t.hour * 3600 + t.minute * 60 + t.second)).to_numpy()

    for well in wells:
        params_as_vars = np.zeros(6)  # Updated to match new param count
        y_vals = raw_df[well].to_numpy()

        ir = IsotonicRegression(increasing=True)
        x_fit = ir.fit_transform(time_column, y_vals)
        total_yield = max(y_vals)
        slopes = np.diff(x_fit) / np.diff(time_column)
        t0 = max(slopes)

        # Fit and filter only linear and dec_exp
        params, scores = least_squares_fitter(time_column, x_fit)
        filtered_params = {k: v for k, v in params.items() if k in ['linear', 'dec exp']}
        filtered_scores = [scores[i] for i, k in enumerate(params.keys()) if k in ['linear', 'dec exp']]
        best_fit = list(filtered_params.keys())[np.argmax(filtered_scores)]
        
        params_as_vars[params_indices[best_fit]] = filtered_params[best_fit]

        max_react_rates.append(t0)
        yields.append(total_yield)
        best_fits.append(best_fit)
        all_params.append(params_as_vars)
        all_params_lite.append(filtered_params[best_fit])
        x_fits.append(x_fit)

    max_react_rates = np.array(max_react_rates)
    if pos_wells:
        pos_well_ind = [i for i, col in enumerate(wells) if col in pos_wells]
        pos_wells_t0 = max_react_rates[pos_well_ind].mean()
        norm_react_rates = max_react_rates / pos_wells_t0
    else:
        norm_react_rates = max_react_rates

    if return_coef:
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

class BayesianOptimization:
    """
    Bayesian optimization.

    BayesianOptimization fits a Gaussian process regressor to suggest
    candidate points for a following iteration. It does this as a middle-man
    between NumPy and PyTorch while attempting to resemble sklearn regressors
    in terms of user interaction.

    Attributes
    ----------
    gp_mean
    gp_cov
    gp
    ucb_hyperparam
    num_restarts
    raw_samples
    var_names
    train_x
    train_y

    """

    def __init__(self):
        self.gp_mean = None
        self.gp_cov = None
        self.gp = None
        self.ucb_hyperparam = 0.1
        self.num_restarts = 5
        self.raw_samples = 20

    def _build_model(self, train_x, train_y, believer_mode=False):
        self.partitioning = 0
        if len(self.y_names) > 1:
            self.partitioning = DominatedPartitioning(
                ref_point=torch.tensor(self.ref_point),
                Y=train_y)

        gp_dict = {'Single-Task GP':[SingleTaskGP,
                            {'covar_module':None}],
            'Mixed Single-Task GP':[MixedSingleTaskGP,
                            {'cat_dims':self.cat_dims, 'cont_kernel_factory':None}],
            'SAASBO':[general_saasbo_gp, dict()]}

        if believer_mode is False:
            self.gp = gp_dict[self.model_type][0](train_x, train_y, **gp_dict[self.model_type][1])
            self.backup_gp = gp_dict[self.model_type][0](train_x, train_y, **gp_dict[self.model_type][1])
        else:
            self.gp = gp_dict[self.model_type][0](train_x, train_y, **gp_dict[self.model_type][1])

    def fit(self, X, y, optim_direc=None, cat_dims=None, model_type=None,kernel=None):
        """
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features) of training data.

        y : list of strings of the names corresponding to the columns for target data found in X.

        optim_direc : list of strings or weights as floats with len(y). strings should be 'min' or 'max' to show whether we minimize or maximize this target. If left empty, all targets will be maximized.

        cat_dims : list of names corresponding to the columns of the input X that should be considered categorical features.
        
        Currently accounts for mixed spaces and continuous spaces.
        """
        self.optim_direc = optim_direc
        train_x = X.drop(y, axis=1)
        self.var_names = list(train_x.columns)
        train_y = X[y]
        if optim_direc:
            weights = [
                1 if val == "max" else -1 if val == "min" else val
                for val in optim_direc
            ]
            train_y = train_y.mul(weights, axis='columns')

        if (model_type is None or model_type == 'Mixed Single-Task GP') and cat_dims:
            model_type = 'Mixed Single-Task GP'
            cat_dims = [self.var_names.index(v) for v in cat_dims]
        elif model_type != 'Mixed Single-Task GP' and cat_dims:
            dummies = pd.get_dummies(X[cat_dims], columns=cat_dims).astype(int)
            train_x = pd.concat([X.drop(columns=cat_dims), dummies], axis=1)
            self.var_names = list(train_x.columns)
            cat_dims = [self.var_names.index(col) for col in dummies.columns]
        else:
            model_type = 'Single-Task GP'

        train_x = train_x.to_numpy().reshape(-1,np.shape(train_x)[1])
        train_y = train_y.to_numpy().reshape(-1,np.shape(train_y)[1])
        self.ref_point = train_y.min(axis=0)
        self.train_x = torch.tensor(train_x)
        self.train_y = torch.tensor(train_y)
        self.y_names = y
        self.model_type = model_type
        self.cat_dims = cat_dims
        self._build_model(self.train_x, self.train_y)

        return self
    
    def _acqf_optimizer(self, train_x, train_y, q, bounds, believer_mode=False):
        acq_dict = {
        'LogEI': [LogExpectedImprovement, {'best_f':train_y.max()}],
        'UCB': [UpperConfidenceBound, {'beta':self.ucb_hyperparam}],
        'LogPI': [ProbabilityOfImprovement, {'best_f':train_y.max()}],
        'qLogEI': [qLogExpectedImprovement, {'best_f':train_y.max()}],
        'qUCB': [qUpperConfidenceBound, {'beta':self.ucb_hyperparam}],
        'qLogPI': [qProbabilityOfImprovement, {'best_f':self.train_y.max()}],
        'EHVI' : [ExpectedHypervolumeImprovement, {'ref_point': self.ref_point,
                                                    'partitioning': self.partitioning}],
        'qEHVI' : [qExpectedHypervolumeImprovement, {'ref_point': self.ref_point,
                                                    'partitioning': self.partitioning}]
        }

        acq_func = acq_dict[self.acq_func_name][0](self.gp, **acq_dict[self.acq_func_name][1])

        if bounds:
            self.bounds = torch.tensor(bounds)
        else:
            bounds = (torch.min(train_x, 0)[0], torch.max(train_x, 0)[0])
            self.bounds = torch.stack(bounds)

        candidate, _ = optimize_acqf(acq_func, bounds=self.bounds, q=q, 
                                             num_restarts=self.num_restarts, raw_samples=self.raw_samples)
        if believer_mode is False:
            self.acq_func = acq_func
            self.backup_acq_func = acq_func
        else:
            self.acq_func = acq_func
        return candidate, _

    def _predict(self, X):
        if (len(self.y_names) == 1 and self.model_type == 'SAASBO'):
            prediction = self.gp.posterior(X).mean.mean(dim=0)
        elif (len(self.y_names) > 1 and self.model_type == 'SAASBO'):
            preds = []
            for model_ind in range(len(self.y_names)):
                pred = self.gp.models[model_ind].posterior(X).mean.mean(dim=0)
                preds.append(pred)
            prediction = torch.cat(preds, dim=1)
        else:
            prediction = self.gp.posterior(X).mean
        return prediction
        
    def candidates(self, q, acq_func_name=None, bounds=None, export_df=False, q_sampling_method=None):
        """
        Optimizes an acquisition function to return candidates
        q_sampling_method : If None, Monte Carlo will be chosen if q>1 
        and the analytic method will be chosen for q=1. ["Monte Carlo", "Believer", "Thompson"]
        """ 

        if acq_func_name is None and len(self.y_names) > 1:
            acq_func_name = 'EHVI'
        elif acq_func_name is None and len(self.y_names) == 1:
            acq_func_name = 'LogEI'

        analytic_iter_n = 1
        if q_sampling_method is None and q > 1:
            acq_func_name = 'q' + acq_func_name
        elif q_sampling_method == "Monte Carlo":
            acq_func_name = 'q' + acq_func_name
        elif (q_sampling_method == "Believer" and q > 1):
            analytic_iter_n = q - 1
            q = 1
        self.acq_func_name = acq_func_name

        candidate, _ = self._acqf_optimizer(self.train_x, self.train_y, q, bounds)
        prediction = self._predict(candidate)

        if q_sampling_method=="Believer":
            all_candidates = candidate
            all_predictions = prediction
            for _ in range(analytic_iter_n):
                retrain_y = torch.cat((self.train_y, prediction))
                retrain_x = torch.cat((self.train_x, candidate))
                self._build_model(retrain_x, retrain_y, believer_mode=True)
                candidate, _ = self._acqf_optimizer(retrain_x, retrain_y, q, bounds, believer_mode=True)
                prediction = self._predict(candidate)

                all_candidates = torch.cat((all_candidates, candidate))
                all_predictions = torch.cat((all_predictions, prediction))
            candidate, prediction = all_candidates, all_predictions

        candidate, prediction = candidate.detach().numpy(), prediction.detach().numpy()
        if export_df:
            pred_df = pd.DataFrame(prediction, columns=self.y_names)
            candidate_df = pd.DataFrame(candidate, columns=self.var_names)
            suggested_df = pd.concat((candidate_df, pred_df),axis=1)
            return suggested_df
        else:
            return candidate, prediction

    #def visualize(black_box_f=None, projection='2D', return_frame=False):