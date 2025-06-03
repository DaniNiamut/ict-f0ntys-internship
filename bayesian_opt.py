import torch
from botorch.models import SingleTaskGP, MixedSingleTaskGP, SaasFullyBayesianSingleTaskGP, ModelListGP
from botorch.acquisition.analytic import (LogExpectedImprovement, UpperConfidenceBound,
                                        ProbabilityOfImprovement)
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
#Would have included thompson, but it is not an operational feature in BoTorch yet.
#from botorch.acquisition.objective import PosteriorTransform
#from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from cost_aware_acquisition import (ingredient_cost, AnalyticAcquisitionFunctionWithCost,
                                    MCAcquisitionFunctionWithCost, ExpectedHypervolumeImprovementWithCost,
                                    qExpectedHypervolumeImprovementWithCost)
from botorch.optim import optimize_acqf
import numpy as np
from botorch import fit_fully_bayesian_model_nuts
import pandas as pd
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
import warnings
from bodi import HammingEmbeddingDictionary
#from matplotlib import pyplot as plt

from botorch.exceptions import InputDataWarning
# Ignore only the scaling warning
warnings.filterwarnings("ignore", category=InputDataWarning)

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

def reverse_one_hot(arr, var_names, cat_cols, var_names_original, **kwargs):
    df = pd.DataFrame(arr, columns=var_names)

    for cat in cat_cols:
        cat_cols = [col for col in var_names if col.startswith(f"{cat}_")]
        df[cat] = df[cat_cols].idxmax(axis=1).str.replace(f"{cat}_", "")
        df.drop(columns=cat_cols, inplace=True)
    # Reorder columns to match original
    df = df[[col for col in var_names_original if col in df.columns]]
    return df

def snap_categories(arr, var_names, cat_vals, **kwargs):
    is_df = isinstance(arr, pd.DataFrame)
    data = pd.DataFrame(arr, columns=var_names)

    for col, valid_vals in cat_vals.items():
        #maybe this'll fix the trunk problem?
        data[col] = data[col].apply(lambda x, current_valid_vals=valid_vals:
                                    min(current_valid_vals, key=lambda v: abs(x - v)))

    return data if is_df else data.to_numpy()

def df_reordering(arr, var_names_original, **kwargs):
    return arr[[col for col in var_names_original
                                             if col in arr.columns]]

class FeaturePreprocessor:
    def _set_up_feature_processing(self, train_x, train_y, optim_direc, cat_dims):
        if optim_direc:
            weights = [1 if val == "max" else -1 if val == "min" else val for val in optim_direc]
            train_y = train_y.mul(weights, axis='columns')
        self.cat_cols = cat_dims
        self.cat_classes = None
        if cat_dims is not None:
            self.cat_vals = {col: sorted(train_x[col].unique().tolist()) for col in cat_dims}
            cat_dict_keys = list(self.cat_vals.keys())
            self.cat_classes = [int(max(self.cat_vals[key])) for key in cat_dict_keys]
        return train_y

    def _setup_model_and_clean_up_method(self, train_x, cat_dims, model_type):
        self.input_transform = None
        if (model_type is None or model_type == 'Mixed Single-Task GP') and cat_dims:
            model_type = 'Mixed Single-Task GP'
            cat_dims = [self.var_names.index(v) for v in cat_dims]
            self.clean_up_method = snap_categories
        if model_type == 'HED' and cat_dims:
            cat_dims = [self.var_names.index(v) for v in cat_dims]
            self.input_transform = HammingEmbeddingDictionary(cat_dims=cat_dims,
                                      reduced_cat_dim=self.dictionary_m,
                                      classes_per_cat=self.cat_classes)
        elif model_type != 'Mixed Single-Task GP' and cat_dims:
            dummies = pd.get_dummies(train_x[cat_dims], columns=cat_dims).astype(int)
            train_x = pd.concat([train_x.drop(columns=cat_dims), dummies], axis=1)
            self.var_names = list(train_x.columns)
            self.clean_up_method = reverse_one_hot
        else:
            model_type = 'Single-Task GP'
        return train_x, cat_dims, model_type

    def _build_dicts(self):
        self.gp_dict = {
            'Single-Task GP': [SingleTaskGP, {'covar_module': None}],
            'Mixed Single-Task GP': [MixedSingleTaskGP, {'cat_dims': self.cat_dims, 'cont_kernel_factory': None}],
            'SAASBO': [general_saasbo_gp, dict()],
            'HED': [SingleTaskGP, {'input_transform':self.input_transform}]
        }

        self.acq_dict = {
            'LogEI': [LogExpectedImprovement, {'best_f': self.train_y.max()}, AnalyticAcquisitionFunctionWithCost],
            'UCB': [UpperConfidenceBound, {'beta': self.ucb_hyperparam}, AnalyticAcquisitionFunctionWithCost],
            'LogPI': [ProbabilityOfImprovement, {'best_f': self.train_y.max()}, AnalyticAcquisitionFunctionWithCost],
            'qLogEI': [qLogExpectedImprovement, {'best_f': self.train_y.max()}, MCAcquisitionFunctionWithCost],
            'qUCB': [qUpperConfidenceBound, {'beta': self.ucb_hyperparam}, MCAcquisitionFunctionWithCost],
            'qLogPI': [qProbabilityOfImprovement, {'best_f': self.train_y.max()}, MCAcquisitionFunctionWithCost],
            'EHVI': [ExpectedHypervolumeImprovement,
                     {'ref_point': self.ref_point, 'partitioning': self.partitioning},
                     ExpectedHypervolumeImprovementWithCost],
            'qEHVI': [qExpectedHypervolumeImprovement,
                      {'ref_point': self.ref_point, 'partitioning': self.partitioning},
                      qExpectedHypervolumeImprovementWithCost]
        }

class AcquisitionHandler:

    def __init__(self):
        self.acq_func_name = None
        self.y_names = None

    def _acqf_optimizer(self, train_x, q, bounds, believer_mode=False, input_weights=None):
        acq_func = self.acq_dict[self.acq_func_name][0](self.gp, **self.acq_dict[self.acq_func_name][1])
        if input_weights is not None:
            cost_model = ingredient_cost(weights=input_weights, fixed_cost=1e-5)
            acq_func = self.acq_dict[self.acq_func_name][2](self.gp, acq_func, cost_model)

        if bounds:
            self.bounds = torch.tensor(bounds)
        else:
            bounds = (torch.min(train_x, 0)[0], torch.max(train_x, 0)[0])
            self.bounds = torch.stack(bounds)

        candidate, _ = optimize_acqf(acq_func, bounds=self.bounds, q=q,
                                     num_restarts=self.num_restarts, raw_samples=self.raw_samples)

        if not believer_mode:
            self.acq_func = acq_func
            self.backup_acq_func = acq_func
        else:
            self.acq_func = acq_func
        return candidate, _

    def _acq_func_determiner(self, acq_func_name, q_sampling_method, q):
        if acq_func_name is None and len(self.y_names) > 1:
            acq_func_name = 'EHVI'
        elif acq_func_name is None and len(self.y_names) == 1:
            acq_func_name = 'LogEI'

        analytic_iter_n = None
        if q_sampling_method is None and q > 1:
            acq_func_name = 'q' + acq_func_name
        elif q_sampling_method == "Monte Carlo":
            acq_func_name = 'q' + acq_func_name
        elif q_sampling_method == "Believer":
            analytic_iter_n = q - 1
            q = 1

        self.acq_func_name = acq_func_name
        return q, analytic_iter_n

    def _believer_update(self, candidate, prediction, analytic_iter_n, bounds, input_weights):
        believer_iter = 0
        all_candidates = candidate
        all_predictions = prediction
        retrain_y = self.train_y
        retrain_x = self.train_x

        while believer_iter < analytic_iter_n:
            retrain_y = torch.cat((retrain_y, prediction))
            retrain_x = torch.cat((retrain_x, candidate))
            self._build_model(retrain_x, retrain_y, believer_mode=True)
            candidate, _ = self._acqf_optimizer(train_x=retrain_x, q=1, bounds=bounds,
                                                believer_mode=True, input_weights=input_weights)
            candidate = candidate.round(decimals=2)
            if not torch.any(torch.all(candidate == all_candidates, dim=1)):
                prediction, _ = self._predict(candidate)
                all_candidates = torch.cat((all_candidates, candidate))
                all_predictions = torch.cat((all_predictions, prediction))
                believer_iter += 1
        return all_candidates, all_predictions

class BayesianOptimization(FeaturePreprocessor, AcquisitionHandler):
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
        self.dictionary_m = 128

        self.cat_vals = None

    def _build_model(self, train_x, train_y, believer_mode=False):
        self.partitioning = 0
        if len(self.y_names) > 1:
            self.partitioning = DominatedPartitioning(
                ref_point=torch.tensor(self.ref_point),
                Y=train_y)
        self._build_dicts()
        if believer_mode is False:
            self.gp = self.gp_dict[self.model_type][0](train_x, train_y, **self.gp_dict[self.model_type][1])
            self.backup_gp = self.gp_dict[self.model_type][0](train_x, train_y, **self.gp_dict[self.model_type][1])
        else:
            self.gp = self.gp_dict[self.model_type][0](train_x, train_y, **self.gp_dict[self.model_type][1])

    def _predict(self, X):
        if (len(self.y_names) == 1 and self.model_type == 'SAASBO'):
            prediction = self.gp.posterior(X).mean.mean(dim=0)
            variance = self.gp.posterior(X).variance.mean(dim=0)
        elif (len(self.y_names) > 1 and self.model_type == 'SAASBO'):
            preds, variances = [], []
            for model_ind in range(len(self.y_names)):
                pred = self.gp.models[model_ind].posterior(X).mean.mean(dim=0)
                variance = self.gp.models[model_ind].posterior(X).variance.mean(dim=0)
                variances.append(variance)
                preds.append(pred)
            prediction = torch.cat(preds, dim=1)
            variance = torch.cat(variances, dim=1)
        else:
            prediction = self.gp.posterior(X).mean
            variance = self.gp.posterior(X).variance
        return prediction, variance

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
        self.clean_X_cols = list(X.columns)
        self.var_names = list(train_x.columns)
        train_y = X[y]
        train_y = self._set_up_feature_processing(train_x, train_y, optim_direc, cat_dims)
        self.clean_up_method = df_reordering

        train_x, cat_dims, model_type = self._setup_model_and_clean_up_method(train_x, cat_dims, model_type)
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

    def candidates(self, q, acq_func_name=None, bounds=None, export_df=False,
                   q_sampling_method=None, input_weights=None):
        """
        Optimizes an acquisition function to return candidates
        q_sampling_method : If None, Monte Carlo will be chosen if q>1 
        and the analytic method will be chosen for q=1. ["Monte Carlo", "Believer"]
        """
        q, analytic_iter_n = self._acq_func_determiner(acq_func_name=acq_func_name,
                                                       q_sampling_method=q_sampling_method, q=q)
        candidate, _ = self._acqf_optimizer(train_x=self.train_x, q=q,
                                            bounds=bounds, input_weights=input_weights)
        candidate = candidate.round(decimals=2)
        prediction, _ = self._predict(candidate)

        if q_sampling_method=="Believer":
            candidate, prediction = self._believer_update(candidate=candidate, prediction=prediction,
                                                          analytic_iter_n=analytic_iter_n, bounds=bounds,
                                                          input_weights=input_weights)
        candidate, prediction = candidate.detach().numpy(), prediction.detach().numpy()

        if export_df:
            pred_df = pd.DataFrame(prediction, columns=self.y_names)
            candidate_df = pd.DataFrame(candidate, columns=self.var_names)
            suggested_df = pd.concat((candidate_df, pred_df),axis=1)
            suggested_df = self.clean_up_method(arr=suggested_df, var_names=list(suggested_df.columns),
                            cat_vals=self.cat_vals, cat_cols=self.cat_cols,
                            var_names_original=self.clean_X_cols)
            return suggested_df
        else:
            return candidate, prediction

    #def visualize(bounds=None):
