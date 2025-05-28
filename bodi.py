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
    Function taken from Huawei-.. mcbo GitHub
    """
    sorted_samples = np.random.uniform(0, 1, (n_samples, d + 1))
    sorted_samples[:, 0] = 0
    sorted_samples[:, -1] = 1
    sorted_samples.sort(-1)
    return torch.tensor(sorted_samples[:, 1:] - sorted_samples[:, :-1])

def diverse_random_dict_sample(m: int, n_cats_per_dim: List[int]) -> Tensor:
    """
    Function heavily inspired by Huawei-... GitHub version.
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
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor) -> Tensor:
        ctx.save_for_backward(u, v)
        return torch.mean((u != v).double())

    @staticmethod
    def backward(ctx, grad_output):
        u, v = ctx.saved_tensors
        #incorrect gradients
        return grad_output * u, grad_output * v

def hed_transform(A, X):
    observations, _ = X.shape
    reduced_dim, _ = A.shape
    dim_reduc_x = torch.zeros((observations, reduced_dim))
    for observation in range(observations):
        for m in range(reduced_dim):
            dim_reduc_x[observation, m] = HammingDistance.apply(A[m,:], X[observation,:])
    return dim_reduc_x

class HammingEmbeddingDictionary(InputTransform):
    r"""Abstract base class for input transforms.

    Properties:
        is_one_to_many: A boolean denoting whether the transform produces
            multiple values for each input.
        transform_on_train: A boolean indicating whether to apply the
            transform in train() mode.
        transform_on_eval: A boolean indicating whether to apply the
            transform in eval() mode.
        transform_on_fantasize: A boolean indicating whether to apply
            the transform when called from within a `fantasize` call.
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
        approximate: bool = False,
        tau: float = 1e-3,
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
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n' x d`-dim tensor of transformed inputs.
        """
        if self.training:
            if self.transform_on_train:
                return self.transform(X)
        elif self.transform_on_eval:
            if fantasize.off() or self.transform_on_fantasize:
                return self.transform(X)

    def transform(self, X: Tensor) -> Tensor:
        
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        cont_dims = [i for i in range(X.shape[-1]) if i not in self.cat_dims]
        if len(X.shape) > 2:
            X_cat = X[:, :, self.cat_dims]
            X_cont = X[:, :, cont_dims]
            samples, candidates = X_cat.shape[0], X_cat.shape[1]
            X_cat_red_dim = torch.zeros(samples, candidates, self.reduced_cat_dim)
            for i in range(samples):
                X_cat_red_dim[i,:,:] = hed_transform(self.dictionary, X_cat[i,:,:])
            X_red_dim = torch.cat((X_cat_red_dim, X_cont), axis=2)
        else:
            X_cat = X[:, self.cat_dims]
            X_cont = X[:, cont_dims]
            X_cat_red_dim = hed_transform(self.dictionary, X_cat)
            X_red_dim = torch.cat((X_cat_red_dim, X_cont), axis=1)
        return X_red_dim