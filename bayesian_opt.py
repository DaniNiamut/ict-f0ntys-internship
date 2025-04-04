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
        '''
        Optimizes an acquisition function to return candidates
        '''
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






