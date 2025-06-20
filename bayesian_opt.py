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
from botorch.optim import optimize_acqf, optimize_acqf_mixed
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

def general_saasbo_gp(X : torch.Tensor, Y : torch.Tensor) -> ModelListGP:
    """
    Creates a model list of SAASBO GPs for multiple outputs.

    Parameters
    ----------
    X : Tensor
        Input data with shape (n_samples, n_features).

    Y : Tensor
        Output data with shape (n_samples, n_outputs).

    Returns
    -------
    gp : ModelListGP instance
        Contains SAASBO GPs for each output in Y.
    """
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

def snap_categories(
    arr : np.ndarray,
    var_names : list[str],
    cat_vals : dict[str ,
    list[float]],
    **kwargs
    ) -> np.ndarray | pd.DataFrame:
    """
    Puts (snaps) each categorical variable in arr to the closest valid category in cat_vals per categorical variable.

    Parameters  
    ----------
    arr : Numpy array or Dataframe
        Input data with shape (n_samples, n_features) to be snapped to valid categories.

    var_names : List of Strings
        Names of the variables in the array. var_names has to be the same length as n_features in arr.

    cat_vals : Dictionary
        The keys are the names of each catregory and the values are lists of valid values for that category. 

    Returns
    -------
    Pandas Dataframe or Numpy array
        Same shape as arr, but with categorical variables snapped to the closest valid category value.
    """
    is_df = isinstance(arr, pd.DataFrame)
    data = pd.DataFrame(arr, columns=var_names)

    for col, valid_vals in cat_vals.items():
        # Chooses minimum distance between valid categories (valid_vals) and the actual candidate value (X).
        data[col] = data[col].apply(lambda x, current_valid_vals=valid_vals:
                                    min(current_valid_vals, key=lambda v: abs(x - v)))
    
    data = data.astype(float)
    return data if is_df else data.to_numpy()

def reverse_one_hot(
    arr : np.ndarray | pd.DataFrame ,
    var_names : list[str],
    cat_cols : list[str],
    var_names_original : list[str], **kwargs
    )-> pd.DataFrame:
    """
    Performs reverse one-hot encoding on the given array or dataframe.

    Parameters
    ----------
    arr : Numpy array or Dataframe
        Input data with shape (n_samples, n_features) to be reversed from one-hot encoding to original categories.

    var_names : List of Strings
        Names of the variables in the array. var_names has to be the same length as n_features in arr.

    cat_cols : List of Strings
        Names of the categorical columns.

    var_names_original : List of Strings
        Containing the original order of the variable names.

    Returns
    -------
    df : Pandas DataFrame
        Contains categorical variables reversed from one-hot encoding to their original categories.
    """
    df = pd.DataFrame(arr, columns=var_names)

    for cat in cat_cols:
        prefix = f"{cat}_"
        matching_cols = [col for col in df.columns if col.startswith(prefix)]

        if matching_cols:
            df[cat] = df[matching_cols].idxmax(axis=1).str.replace(prefix, "", regex=False)
            df.drop(columns=matching_cols, inplace=True)

    final_cols = [col for col in var_names_original if col in df.columns]
    df = df[final_cols]
    return df

def snap_and_reverse_one_hot(
    arr: np.ndarray | pd.DataFrame,
    var_names: list[str],
    cat_vals: dict[str, list[float]],
    cat_cols: list[str],
    var_names_original: list[str]
) -> pd.DataFrame:
    """
    Applies category snapping followed by reverse one-hot encoding.

    Parameters
    ----------
    arr : Numpy array or Dataframe
        Input data with shape (n_samples, n_features) of input data to be snapped to valid categories and reversed from one-hot encoding.

    var_names : List of Strings
        Names of the variables in the array. var_names has to be the same length as n_features in arr.

    cat_vals : dictionary
        The keys are the names of each catregory and the values are lists of valid values for that category. 

    cat_cols : list of strings
        Names of the categorical columns.

    var_names_original : list of strings
        Containing the original order of the variable names.

    Returns
    -------
    df : Pandas DataFrame
        DataFrame with categories snapped and one-hot columns reversed.
    """
    snapped = snap_categories(
        arr=arr,
        var_names=var_names,
        cat_vals=cat_vals
    )

    return reverse_one_hot(
        arr=snapped,
        var_names=var_names,
        cat_cols=cat_cols,
        var_names_original=var_names_original
    )

def df_reordering(
    arr : np.ndarray,
    var_names_original : list[str],
    **kwargs
    ) -> pd.DataFrame | np.ndarray:
    """
    Reorders the columns of a dataframe or numpy array to match the original order of variable names.

    Parameters
    ----------
    arr : numpy array or dataframe
        Input data with shape (n_samples, n_features) to be reordered.

    var_names_original : list of strings
        Containing the original order of the variable names.

    Returns
    -------
    arr : Pandas DataFrame or Numpy array
        Contains the columns reordered to match the original order of variable names.
    """
    return arr[[col for col in var_names_original
                                             if col in arr.columns]]

class FeaturePreprocessor:
    """
    A class used to preprocess features, especially handling categorical inputs,
    and set up models, acquisition functions, and optimizers for Bayesian Optimization.

    Attributes
    ----------
    cat_cols : list of strings or None
        Names of categorical columns in the training data.

    cat_vals : dictionary or None
        Dictionary mapping each categorical column to its sorted unique values.

    cat_dict_keys : list of strings
        List of keys (column names) from `cat_vals`.

    cat_classes : list of integers or None
        The number of classes per categorical variable (used for embedding or one-hot encoding).

    input_transform : callable or None
        If using embeddings (e.g., Hamming Embedding Dictionary), this is the transform applied to the inputs.

    clean_up_method : callable
        Method used to clean up or reverse transformations (e.g., reverse one-hot encoding, snapping categories).

    var_names : list of strings
        Column names of the transformed training data. Helps track features after transformations.

    ref_point : Tensor
        Reference point for hypervolume calculations in multi-objective optimization.

    partitioning : DominatedPartitioning or None
        Used for calculating hypervolume improvement in multi-objective acquisition functions.

    fixed_features_list : list of dictionries or None
        List of fixed feature combinations for categorical dimensions used in the optimizer.

    gp_dict : dictionary
        Dictionary mapping model types to [GP model class, kwargs] for instantiation.

    acq_dict : dictionary
        Dictionary mapping acquisition function names to [class, kwargs, cost-aware wrapper].

    optimizer_dict : dictionary
        Dictionary mapping optimizer strategies to [optimizer function, kwargs].
    """
    def _set_up_feature_processing(
        self,
        train_x : torch.Tensor,
        train_y : torch.Tensor,
        optim_direc : list[str | float] = None,
        cat_dims : list[str] = None
        ) -> torch.Tensor:
        """
        Sets up the feature processing for the training data.

        Attributes
        ----------
        train_x : Tensor
            Training data with shape (n_samples, n_features).

        train_y : Tensor
            Target data with shape (n_samples, n_targets).

        optim_direc : list of strings or weights as floats with len(y). 
            Strings should be "min" or "max" to show whether we minimize or maximize this target. If left empty, all targets will be maximized.
            Example: ["min"],["max"], [-1(any negative number for minimizing)] or [1(any positive number for maximizing)] for a single objective optimization problem, 
            ["max", "min"] or [-1, 1] for a multi-objective optimization problem with two objectives, etc.

        cat_dims : list of strings
            Names corresponding to the columns of the input X that should be considered categorical features.
            If None, no categorical features will be considered.

        Returns
        -------
        train_y : Tensor
            Target data with shape (n_samples, n_targets) after processing.
        """
        if optim_direc:
            weights = [1 if val == "max" else -1 if val == "min" else val for val in optim_direc]
            train_y = train_y.mul(weights, axis="columns")
        self.cat_cols = cat_dims
        self.cat_classes = None
        if cat_dims is not None:
            self.cat_vals = {col: sorted(train_x[col].unique().tolist()) for col in cat_dims}
            self.cat_dict_keys = list(self.cat_vals.keys())
            self.cat_classes = [int(max(self.cat_vals[key])) for key in self.cat_dict_keys]
        return train_y

    def _setup_model_and_clean_up_method(self, train_x : torch.Tensor, cat_dims : list[str], model_type : str) -> tuple[torch.Tensor, list[int], str]:
        """
        Sets up the model type and the clean-up method based on the input data and categorical dimensions.

        Parameters
        ----------
        train_x : Tensor
            Training data with shape (n_samples, n_features).

        cat_dims : list of strings
            Names corresponding to the columns of the input X that should be considered categorical features.

        model_type : string
            Can be one of ["Single-Task GP", "Mixed Single-Task GP", "SAASBO", "HED"].

        Returns
        -------
        train_x : Tensor
            Training data with shape (n_samples, n_features) after processing.

        cat_dims : list of integers
            Indices of the categorical dimensions in the train_x.

        model_type : string
            The model type to use for the Gaussian process.
        """
        self.input_transform = None
        if (model_type is None or model_type == "Mixed Single-Task GP") and cat_dims:
            model_type = "Mixed Single-Task GP"
            cat_dims = [self.var_names.index(v) for v in cat_dims]
            self.clean_up_method = snap_categories
        if model_type == "HED" and cat_dims:
            cat_dims = [self.var_names.index(v) for v in cat_dims]
            self.input_transform = HammingEmbeddingDictionary(cat_dims=cat_dims,
                                      reduced_cat_dim=self.dictionary_m,
                                      classes_per_cat=self.cat_classes)
            self.clean_up_method = snap_categories
        elif model_type != "Mixed Single-Task GP" and cat_dims:
            multi_cat_dims = [col for col in cat_dims if train_x[col].nunique() > 2]
            binary_cat_dims = [col for col in cat_dims if train_x[col].nunique() == 2]
            dummies = pd.DataFrame(index=train_x.index)
            if multi_cat_dims:
                dummies = pd.get_dummies(train_x[multi_cat_dims], columns=multi_cat_dims)
                train_x = pd.concat([train_x.drop(columns=multi_cat_dims), dummies], axis=1).astype(float)
            self.var_names = list(train_x.columns)
            cat_dims = list(dummies.columns) + binary_cat_dims
            self.cat_vals = {col: sorted(train_x[col].unique().tolist()) for col in cat_dims}
            self.cat_dict_keys = list(self.cat_vals.keys())
            self.cat_classes = [int(max(self.cat_vals[key])) for key in self.cat_dict_keys]
            cat_dims = [self.var_names.index(v) for v in cat_dims]
            self.clean_up_method = reverse_one_hot
        else:
            model_type = "Single-Task GP"
        return train_x, cat_dims, model_type

    def _dict_setup(self):
        """
        Support function that calculates values to be used in the _build_dicts function.
        """
        self.ref_point,_ = torch.min(self.train_y, 0)
        self.partitioning = None
        if len(self.y_names) > 1:
            self.partitioning = DominatedPartitioning(
                ref_point=self.ref_point,
                Y=self.train_y)

        self.fixed_features_list = None
        if self.cat_dims is not None:
            if len(self.cat_dims) < self.max_cat_dims:
                all_cats = tuple(self.cat_vals.values())
                all_cat_permutations = np.array(np.meshgrid(*all_cats)).T.reshape(-1, len(self.cat_dims))
                all_cat_permutations = all_cat_permutations.tolist()
                self.fixed_features_list = [dict(zip(self.cat_dims, permutation)) for permutation in all_cat_permutations]
            else:
                chosen_permutations = []
                for _ in range(round(len(self.cat_dims) / 2) + self.num_cat_fixed_features):
                    permutation = [np.random.choice(self.cat_vals[dim]) for dim in self.cat_dict_keys]
                    chosen_permutations.append(permutation)
                self.fixed_features_list = [dict(zip(self.cat_dims, permutation)) for permutation in chosen_permutations]
            amount_to_remove = round(len(self.cat_dims) / 2)
            for d in self.fixed_features_list:
                keys = list(d.keys())
                keys_to_remove = np.random.choice(keys, size=amount_to_remove, replace=False)
                for key in keys_to_remove:
                    del d[key]


    def _build_dicts(self):
        """
        Builds dictionaries for the Gaussian process models, acquisition functions, and optimizers.
        It is possible include other BoTorch compliant GP models, acquisition functions, and optimizers by adding them to their respective dictionaries.
        """
        self._dict_setup()

        self.gp_dict = {
            "Single-Task GP": [SingleTaskGP, {"covar_module": None}],
            "Mixed Single-Task GP": [MixedSingleTaskGP, {"cat_dims": self.cat_dims, "cont_kernel_factory": None}],
            "SAASBO": [general_saasbo_gp, dict()],
            "HED": [SingleTaskGP, {"input_transform":self.input_transform}]
        }

        self.acq_dict = {
            # If you want to add more acquisituin functions,you can add them here.
            # Make sure to add them in the correct syntax:
            # "acq_func_name": [AcquisitionFunction class,
            # dictionary containing additional inputs required by the acquistion function class, 
            # AcquisitionFunction class with cost calculated],
            "LogEI": [LogExpectedImprovement, {"best_f": self.train_y.max()}, AnalyticAcquisitionFunctionWithCost],
            "UCB": [UpperConfidenceBound, {"beta": self.ucb_hyperparam}, AnalyticAcquisitionFunctionWithCost],
            "LogPI": [ProbabilityOfImprovement, {"best_f": self.train_y.max()}, AnalyticAcquisitionFunctionWithCost],
            "qLogEI": [qLogExpectedImprovement, {"best_f": self.train_y.max()}, MCAcquisitionFunctionWithCost],
            "qUCB": [qUpperConfidenceBound, {"beta": self.ucb_hyperparam}, MCAcquisitionFunctionWithCost],
            "qLogPI": [qProbabilityOfImprovement, {"best_f": self.train_y.max()}, MCAcquisitionFunctionWithCost],
            "EHVI": [ExpectedHypervolumeImprovement,
                     {"ref_point": self.ref_point, "partitioning": self.partitioning},
                     ExpectedHypervolumeImprovementWithCost],
            "qEHVI": [qExpectedHypervolumeImprovement,
                      {"ref_point": self.ref_point, "partitioning": self.partitioning},
                      qExpectedHypervolumeImprovementWithCost]
        }

        self.optimizer_dict = {
            "Multi-Start": [optimize_acqf, dict()],
            "Sequential Fixed Subspace": [optimize_acqf_mixed, {"fixed_features_list": self.fixed_features_list}]
        }


class AcquisitionHandler:
    """
    Determines and optimizes the acquisition function for Bayesian optimization. 
    Also does believer update when required.

    Attributes
    ----------
    acq_func_name : string or None
        Name of the acquisition function to use. 
        If None, will default to "LogEI" for single-objective optimization and "EHVI" for multi-objective optimization.

    y_names : list of strings or None
        Names of the target variables in the training data. 
        Used to determine whether the optimization is single or multi-objective.
    """
    def __init__(self):
        self.acq_func_name = None
        self.y_names = None

    def _acqf_optimizer(
        self,
        train_x : torch.Tensor,
        q : int, bounds,
        believer_mode: bool = False,
        input_weights=None,
        optim_method="Multi-Start", 
        **kwargs
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimizes the acquisition function to return candidates for the next iteration of Bayesian optimization.

        Parameters
        ----------
        train_x : Tensor
            Training data with shape (n_samples, n_features).

        q : integer
            Number of candidates to return.

        bounds : list of 2 lists
            The first contating the lower bounds for each variable the latter the upper bounds for each variable.
            Example: [[1e-5, 1e-5, 1e-5],[14, 1, 2]] for data with 3 features. 

        believer_mode : bool
            If True, will perform the Believer method to update the candidates and predictions.
            Otherwise, no Believer update will be performed.

        input_weights : list of floats
            Weights for each feature in the input space.
            If None, will not use a cost model. 

        optim_method : string
            The optimization method to use for the acquisition function optimization.
            Can be "Multi-Start" or "Sequential Fixed Subspace".

        Returns
        -------
        candidate : Tensor
            Suggested candidates with shape (q, n_features).

        _ : Tensor
            The predicted values for the candidates.
            It is not used in this method, but is returned for consistency with BoTorch's optimize_acqf interface.
        """
        acq_func = self.acq_dict[self.acq_func_name][0](self.gp, **self.acq_dict[self.acq_func_name][1])
        if input_weights is not None:
            cost_model = ingredient_cost(weights=input_weights, fixed_cost=1e-5)
            acq_func = self.acq_dict[self.acq_func_name][2](self.gp, acq_func, cost_model)
        if bounds:
            self.bounds = torch.tensor(bounds)
        else:
            bounds = (torch.min(train_x, 0)[0], torch.max(train_x, 0)[0])
            self.bounds = torch.stack(bounds)

        candidate, _ = self.optimizer_dict[optim_method][0](acq_func, bounds=self.bounds, q=q,
                                     num_restarts=self.num_restarts, raw_samples=self.raw_samples,
                                     **self.optimizer_dict[optim_method][1], **kwargs)

        if not believer_mode:
            self.acq_func = acq_func
            self.backup_acq_func = acq_func
        else:
            self.acq_func = acq_func
        return candidate, _

    def _acq_func_determiner(
        self,
        acq_func_name : str,
        q_sampling_method : str,
        q : int
        ) -> tuple[int, int | None]:
        """
        Determines the acquisition function while takng into account wheter believer uodate is used or not.

        Parameters
        ----------
        q : integer
            Number of candidates to specify whether the acquisition function is single or multi-objective.

        acq_func_name : string
            Name of the acquisition function to use. 
            If None, will default to "LogEI" for single-objective optimization and "EHVI" for multi-objective optimization.

        q_sampling_method : string
            the sampling method to use for the acquisition function optimization.
            Can be "Monte Carlo" or "Believer". If None, will default to "Monte Carlo" if q > 1 and "Believer" if q = 1.
            If "Believer", will return the number of analytic iterations to run for the Believer method.

        Returns
        -------
        q : integer
            Number of candidates to return. If q_sampling_method is "Believer", will be set to 1.

        analytic_iter_n : integer or None
            Number of analytic iterations if Believer method is specified, otherwise None.
        """
        if acq_func_name is None and len(self.y_names) > 1:
            acq_func_name = "EHVI"
        elif acq_func_name is None and len(self.y_names) == 1:
            acq_func_name = "LogEI"

        analytic_iter_n = None
        if q_sampling_method is None and q > 1:
            acq_func_name = "q" + acq_func_name
        elif q_sampling_method == "Monte Carlo":
            acq_func_name = "q" + acq_func_name
        elif q_sampling_method == "Believer":
            analytic_iter_n = q - 1
            q = 1

        self.acq_func_name = acq_func_name
        return q, analytic_iter_n

    def _believer_update(
        self,
        candidate : torch.Tensor,
        prediction : torch.Tensor, analytic_iter_n : int,
        bounds : list[float] = None,
        input_weights : list[float] = None,
        optim_method : str = "Multi-Start"
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the candidates and predictions using the Believer method.
        The Believer method will add new candidates to the existing ones and retrain the model with the new candidates and their predictions.
        The method will keep going until the number of iterations reaches analytic_iter_n(exclusive).

        Parameters
        ----------
        candidate : Tensor
            Current candidates with shape (n_samples, n_features).

        prediction : Tensor
            Predictions for each candidate with shape (n_samples, n_targets).

        analytic_iter_n : integer 
            Number of iterations to run the Believer method for(exclusive).

        bounds : list of 2 lists
            First contating the lower bounds for each variable the latter the upper bounds for each variable.
            Example: [[1e-5, 1e-5, 1e-5],[14, 1, 2]] for data with 3 features.

        input_weights : list of floats
            Weights for each feature in the input space.
            If None, will not use a cost model.

        optim_method : string
            The optimization method to use for the acquisition function optimization.
            Can be "Multi-Start" or "Sequential Fixed Subspace".

        Returns
        -------
        all_candidates : Tensor
            All candidates with shape (n_samples, n_features) suggested so far.

        all_predictions : Tensor
            Predictions of shape (n_samples, n_targets) for each candidate.
        """
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
    gp_mean : Tensor
        Predictive mean with shape (n_samples, n_targets) of the Gaussian Process.

    gp_cov : torch.Tensor
        The predictive covariance shape (n_samples, n_samples, n_targets) of the Gaussian Process.

    gp : BoTorch GP model
        The underlying GP model, e.g., SingleTaskGP, MixedSingleTaskGP, SaasFullyBayesianSingleTaskGP, or ModelListGP.

    ucb_hyperparam : float
        Hyperparameter for the Upper Confidence Bound acquisition function.

    num_restarts : integer
        Number of restarts used during acquisition function optimization.

    raw_samples : integer
        Number of raw samples used during acquisition function optimization.

    dictionary_m : integer
        Number of dictionary atoms used in learned dictionary embedding for categorical features.

    num_cat_fixed_features : integer
        Number of fixed dimensions for categorical features after embedding.

    max_cat_dims : integer
        Maximum dimensionality for categorical features after transformation.

    cat_vals : dictionary or None
        Stores categorical value encodings for inverse transformation.

    cat_dims : list of strings
        Names of input features treated as categorical.

    var_names : list of strrings
        Names of input features used in the model.

    train_x : Tensor
        Tensor of input training data.

    train_y : Tensor
        Tensor of target training data.

    y_names : list of strings
        Names of the target variables.

    model_type : strings
        Type of GP model used ("Single-Task GP", "Mixed Single-Task GP", "SAASBO", or "HED").
    """

    def __init__(self):
        self.gp_mean = None
        self.gp_cov = None
        self.gp = None
        self.ucb_hyperparam = 0.1
        self.num_restarts = 5
        self.raw_samples = 20
        self.dictionary_m = 128
        self.num_cat_fixed_features = 8
        self.max_cat_dims = 5
        self.cat_vals = None

    def _build_model(self, train_x : torch.Tensor, train_y : torch.Tensor, believer_mode: bool = False):
        """
        Builds the Gaussian process model based on the training data and the model type.

        Parameters
        ----------
        train_x : Tensor
            Training data with shape (n_samples, n_features).

        train_y : Tensor
            Target data with shape (n_samples, n_targets).

        believer_mode : bool
            If True, will create a new GP every time a new candidate is suggested.
        """
        self._build_dicts()
        if believer_mode is False:
            self.gp = self.gp_dict[self.model_type][0](train_x, train_y, **self.gp_dict[self.model_type][1])
            self.backup_gp = self.gp_dict[self.model_type][0](train_x, train_y, **self.gp_dict[self.model_type][1])
        else:
            self.gp = self.gp_dict[self.model_type][0](train_x, train_y, **self.gp_dict[self.model_type][1])

    def _predict(self, X : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the mean(actual prediction) and variance of the posterior for the given input points.

        Parameters
        ----------
        X : Tensor
            Input data of shape (n_samples, n_features) for prediction.

        Returns
        -------
        prediction : Tensor
            Predicted values of shape (n_samples, n_targets) for each feature.

        variance : Tensor
            Predicted variances with shape (n_samples, n_targets) for each feature.
        """
        if (len(self.y_names) == 1 and self.model_type == "SAASBO"):
            prediction = self.gp.posterior(X).mean.mean(dim=0)
            variance = self.gp.posterior(X).variance.mean(dim=0)
        elif (len(self.y_names) > 1 and self.model_type == "SAASBO"):
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

    def fit(
        self,
        X : pd.DataFrame,
        y: list[str],
        optim_direc : list[str | float] = None,
        cat_dims : list[str] = None,
        model_type : str = None
        ) -> "BayesianOptimization":
        """
        Fits the BayesianOptimization model to the training data.

        Parameters
        ----------
        X : Pandas Dateaframe
            Training datta with shape (n_samples, n_features).

        y : list of strings
            Names corresponding to the columns for target data found in X.

        optim_direc : list of strings or weights as floats with len(y). 
            Strings should be "min" or "max" to show whether we minimize or maximize this target. If left empty, all targets will be maximized.
            Example: ["min"],["max"], [-1(any negative number for minimizing)] or [1(any positive number for maximizing)] for a single objective optimization problem, 
            ["max", "min"] or [-1, 1] for a multi-objective optimization problem with two objectives, etc.

        cat_dims : list of strings
            Names corresponding to the columns of the input X that should be considered categorical features.

        model_type : string
            Can be one of ["Single-Task GP", "Mixed Single-Task GP", "SAASBO", "HED"]. 
            If None, will default to "Single-Task GP" when no categorical features are present, otherwise will default to "Mixed Single-Task GP".

        Returns
        -------
        self : BayesianOptimization instance
            Fitted with the training data.
        
        Extra Information
        ----------
        Currently accounts for mixed spaces and continuous spaces.
        The kernel is not chosen by the user. The appropriate kernel is picked  depending on the type of model chosen.
         say what type of  kernel is used per model
        """
        self.optim_direc = optim_direc
        train_x = X.drop(y, axis=1)
        self.clean_var_names = list(train_x.columns)
        self.clean_X_cols = list(X.columns)
        self.var_names = list(train_x.columns)
        train_y = X[y]
        train_y = self._set_up_feature_processing(train_x, train_y, optim_direc, cat_dims)
        self.clean_up_method = df_reordering

        train_x, cat_dims, model_type = self._setup_model_and_clean_up_method(train_x, cat_dims, model_type)
        train_x = train_x.to_numpy(dtype=np.float64).reshape(-1,np.shape(train_x)[1])
        train_y = train_y.to_numpy(dtype=np.float64).reshape(-1,np.shape(train_y)[1])
        self.train_x = torch.tensor(train_x)
        self.train_y = torch.tensor(train_y)
        self.y_names = y
        self.model_type = model_type
        self.cat_dims = cat_dims
        self._build_model(self.train_x, self.train_y)

        return self

    def candidates(
        self, q : int,
        acq_func_name: str = None,
        bounds : list[int | float] = None,
        export_df: bool = False,
        q_sampling_method : str = None,
        input_weights : list[float] = None,
        optim_method="Multi-Start"
        ) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
        """
        Optimizes an acquisition function to return candidates for the next iteration of Bayesian optimization.

        Parameters
        ----------
        q : integer
            Number of candidates to return.

        acq_func_name : string
            Name of the acquisition function to use. 
            If None, will default to "LogEI" for single-objective optimization and "EHVI" for multi-objective optimization.

        bounds : list of 2 lists
            The first contating the lower bounds for each variable the latter the upper bounds for each variable.
            Example: [[1e-5, 1e-5, 1e-5],[14, 1, 2]] for data with 3 features.

        export_df : bool
            If True, will return a pandas DataFrame with the suggested candidates and their predidicttions for the target values.

        q_sampling_method : string
            The user can choose between "Monte Carlo" and "Believer".
            If None, will default to "Monte Carlo" if q > 1 and "Believer" if q = 1.

        input_weights : list of floats
            Weights for each feature in the input space.
            If None, will not use a cost model.

        optim_method : string
            The optimization method to use for the acquisition function optimization.
            Can be "Multi-Start" or "Sequential Fixed Subspace".

        Returns
        -------
        If export_df is True:
            Returns a pandas DataFrame with the suggested candidates and their predictions.
        
        Otherwise, returns a tuple of two numpy arrays:
            candidate : numpy array of shape (q, n_features) with the suggested candidates.
            prediction : numpy array of shape (q, n_targets) with the predictions of the features for the suggested candidates.

        Extra Information
        ----------
        It is possible to implement other BoTorch compliant optimizers methods through the _build_dicts function in the FeaturePreprocessor class.
        """
        if (q_sampling_method=="Sequential") and (optim_method=="Multi-Start"):
            seq_dict = {'sequential' : True}
            q_sampling_method = "Monte Carlo"
        else:
            seq_dict = {}
        if optim_method == "Sequential Fixed Subspace" and self.model_type != "Mixed Single-Task GP":
            self.clean_up_method = snap_and_reverse_one_hot
        
        q, analytic_iter_n = self._acq_func_determiner(acq_func_name=acq_func_name,
                                                       q_sampling_method=q_sampling_method, q=q)
        candidate, _ = self._acqf_optimizer(train_x=self.train_x, q=q,
                                            bounds=bounds, input_weights=input_weights,
                                            optim_method=optim_method, **seq_dict)
        candidate = candidate.round(decimals=2)
        prediction, _ = self._predict(candidate)

        if q_sampling_method=="Believer":
            candidate, prediction = self._believer_update(candidate=candidate, prediction=prediction,
                                                          analytic_iter_n=analytic_iter_n, bounds=bounds,
                                                          input_weights=input_weights, optim_method=optim_method)
        candidate, prediction = candidate.detach().numpy(), prediction.detach().numpy()
        
        pred_df = pd.DataFrame(prediction, columns=self.y_names)
        candidate_df = pd.DataFrame(candidate, columns=self.var_names)
        suggested_df = pd.concat((candidate_df, pred_df),axis=1)
        suggested_df = self.clean_up_method(arr=suggested_df, var_names=list(suggested_df.columns),
                        cat_vals=self.cat_vals, cat_cols=self.cat_cols,
                        var_names_original=self.clean_X_cols)
        candidate = suggested_df[self.clean_var_names].to_numpy(dtype=np.float32)
        prediction = suggested_df[self.y_names].to_numpy(dtype=np.float32)
        if export_df:
            return suggested_df
        else:
            return candidate, prediction

    #def visualize(bounds=None):
