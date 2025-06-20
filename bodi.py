from __future__ import annotations

from botorch.models.transforms.input import InputTransform
from torch.autograd import Function
from botorch.models.utils import fantasize
import torch
from torch import Tensor
from typing import List
import numpy as np

def sample_from_simplex(d: int, n_samples: int) -> Tensor:
    """
    Sample uniformly from the interior of a d-dimensional simplex.
    This method is heavily inspired from Huawei's HEBO implementation on GitHub.
    For more details, see: https://github.com/huawei-noah/HEBO/tree/master/MCBO.

    Parameters
    ----------
    d : integer
        The dimension of the simplex.

    n_samples : integer
        The number of samples to draw.

    Returns
    -------
    sorted_samples : Tensor
        A tensor of shape (n_samples, d) containing the sampled points.
        Each row is a point in the d-dimensional simplex.
    """
    sorted_samples = torch.rand((n_samples, d + 1))
    sorted_samples[:, 0] = 0.0
    sorted_samples[:, -1] = 1.0
    sorted_samples, _ = torch.sort(sorted_samples, dim=-1)
    return sorted_samples[:, 1:] - sorted_samples[:, :-1]

def diverse_random_dict_sample(m: int, n_cats_per_dim: List[int]) -> Tensor:
    """
    Creates a diverse random dictionary for categorical variables.
    This method is heavily inspired from Huawei's HEBO implementation on GitHub.
    For more details, see: https://github.com/huawei-noah/HEBO/tree/master/MCBO.

    Parameters
    ----------
    m : integer
        The number of samples to draw.

    n_cats_per_dim : list of integers
        The cardinalities of each categorical variable(dimension).
        For example, if there are 3 categorical variables with 2, 3, and 4 categories respectively,
        n_cats_per_dim would be [2, 3, 4].

    Returns
    -------
    a_dict : Tensor
        A tensor of shape (m, len(n_cats_per_dim)) representing the randomly sampled dictionary.
    """
    a_dict = torch.zeros((m, len(n_cats_per_dim)))
    max_n_cat = max(n_cats_per_dim)
    for i in range(m):
        theta = sample_from_simplex(d=max_n_cat, n_samples=1)[0]
        for j in range(len(n_cats_per_dim)):
            if n_cats_per_dim[j] == max_n_cat:
                subthetas = theta
            else:
                indices = torch.multinomial(input=torch.ones_like(theta), 
                                            num_samples=n_cats_per_dim[j], replacement=False)
                subthetas = theta[indices]
                subthetas /= subthetas.sum()
            a_dict[i, j] = torch.multinomial(input=subthetas, num_samples=1, replacement=False)
    return a_dict

class HammingDistance(Function):
    """
    Hamming distance function using a surrogate gradient for backpropagation. 
    This function computes the Hamming distance between two tensors u and v.
    This class inherits from torch.autograd.Function to define a custom autograd function.
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor) -> Tensor:
        
        ctx.save_for_backward(u, v)
        return (u != v).double().mean(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        u, v = ctx.saved_tensors
        n = u.shape[-1]
        epsilon = 1e-5
        diff = u - v
        grad = diff / (n * diff.abs() + epsilon)
        hamming_grad = grad_output.unsqueeze(-1) * grad
        return hamming_grad, -hamming_grad

def hed_transform(A: Tensor, X: Tensor) -> Tensor:
    """
    Calculates the Hamming distance between each row of A and each row of X.

    Parameters
    ----------
    A : Tensor
        Randomly generated dictionary of shape (m, d) where m is the number of samples and d is the dimension.

    X : Tensor
        Input tensor of shape (n, d) where n is the number of inputs and d is the dimension.

    Returns
    -------
    distances : Tensor
        A tensor of shape (m, n) containing the Hamming distances between each row of A and each row of X.
        Each element (i, j) represents the Hamming distance between the i-th row of A and the j-th row of X.
    """
    A_exp = A.unsqueeze(0).expand(X.size(0), -1, -1)
    X_exp = X.unsqueeze(1).expand(-1, A.size(0), -1)
    distances = HammingDistance.apply(A_exp, X_exp)
    return distances

class HammingEmbeddingDictionary(InputTransform):
    """
    Abstract base class for input transforms.

        Pr:
        is_one_to_many: A boolean denoting whether the transform produces
            multiple values for each input.
        transform_on_train: A boolean indicating whether to apply the
            transform in train() mode.
        transform_on_eval: A boolean indicating whether to apply the
            transform in eval() mode.
        transform_on_fantasize: A boolean indicating whether to apply
            the transform when called from within a `fantasize` call.
    Extra Information
    -----------------
    The names for these atributes in this class are not consistent with their actual use:
    
    """
    
    def __init__(
        self,
        cat_dims: list,
        reduced_cat_dim: int,
        classes_per_cat: list,
        dictionary: Tensor | None = None,
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
    ) -> None:
        super().__init__()
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reduced_cat_dim = reduced_cat_dim
        self.classes_per_cat = classes_per_cat
        self.cat_dims = cat_dims
        if dictionary is None:
            self.dictionary = diverse_random_dict_sample(reduced_cat_dim, classes_per_cat)
        else:
            self.dictionary = dictionary

    def forward(self, X: Tensor) -> Tensor:
        """
        Refer to the BoTorch documentation for the InputTransform class.
        """
        if self.training:
            if self.transform_on_train:
                return self.transform(X)
        elif self.transform_on_eval:
            if fantasize.off() or self.transform_on_fantasize:
                return self.transform(X)

    def transform(self, X: Tensor) -> Tensor:
        """
        Transform the inputs to a model.

        Parameters
        ----------
        X: Tensor
            Input of shape (batch_shape, n, d) or (batch_shape, d) where n is the number of
            input points and d is the number of dimensions.
        
        Returns
        -------
        emb_X: Tensor
            A (batch_shape, n', d')
        """
        cont_dims = [i for i in range(X.shape[-1]) if i not in self.cat_dims]
        if len(X.shape) > 2:
            X_cat = X[:, :, self.cat_dims]
            X_cont = X[:, :, cont_dims]
            batch, n, _ = X_cat.shape

            X_cat_flat = X_cat.reshape(batch * n, -1)

            X_cat_emb_flat = hed_transform(self.dictionary, X_cat_flat)

            X_cat_red = X_cat_emb_flat.view(batch, n, self.reduced_cat_dim)

            emb_x = torch.cat((X_cat_red, X_cont), dim=-1)
        else:
            X_cat = X[:, self.cat_dims]
            X_cont = X[:, cont_dims]
            X_cat_red = hed_transform(self.dictionary, X_cat)
            emb_x = torch.cat((X_cat_red, X_cont), dim=-1)
        return emb_x
