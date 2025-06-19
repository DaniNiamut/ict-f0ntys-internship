from abc import ABC, abstractmethod
import torch
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import SampleReducingMCAcquisitionFunction
from botorch.acquisition.multi_objective.base import (
    MultiObjectiveAnalyticAcquisitionFunction, MultiObjectiveMCAcquisitionFunction
)
from botorch.models.model import Model
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.utils import repeat_to_match_aug_dim
from botorch.utils.transforms import t_batch_mode_transform, concatenate_pending_points
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement

class CostModel(torch.nn.Module, ABC):
    """
    Simple abstract class for a cost model.
    """

    @abstractmethod
    def forward(self, X):
        pass

class ingredient_cost(CostModel):
    def __init__(self, weights, fixed_cost):
        super().__init__()
        self.weights = weights
        self.fixed_cost = fixed_cost
        self.model =  AffineFidelityCostModel(fidelity_weights=weights, fixed_cost=fixed_cost)

    def forward(self, X) -> Tensor:
        return self.model(X)[:, 0]

class AnalyticAcquisitionFunctionWithCost(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x) - c(x), where alpha is a decay
    factor that reduces or increases the emphasis of the cost model c(x).
    """

    def __init__(self, model, acqf, cost_model):
        super().__init__(model=model)
        self.model = model
        self.cost_model = cost_model
        self.acqf = acqf

    def forward(self, X) -> Tensor:
        return self.acqf(X) - self.cost_model(X)[:, 0]
    
class MCAcquisitionFunctionWithCost(SampleReducingMCAcquisitionFunction):

    def __init__(self, model, acqf, cost_model):
    
        super().__init__(model=model)
        self.acqf = acqf
        self.cost_model = cost_model

    def _non_reduced_forward(self, X: Tensor) -> Tensor:
        """Compute the constrained acquisition values at the MC-sample, q level.

        Args:
            X: A `batch_shape x q x d` Tensor of t-batches with `q` `d`-dim
                design points each.

        Returns:
            A Tensor with shape `sample_sample x batch_shape x q`.
        """

        X_flat = X.view(-1, X.size(-1))
        costs_flat = self.cost_model(X_flat) 
        costs = costs_flat.view(X.shape[0], X.shape[1])
        costs = costs.unsqueeze(0).expand(self.sample_shape[0], -1, -1)

        samples, obj = self._get_samples_and_objectives(X)
        obj = obj - costs
        samples = repeat_to_match_aug_dim(target_tensor=samples, reference_tensor=obj)
        acqval = self._sample_forward(obj)  # `sample_sample x batch_shape x q`
        return self._apply_constraints(acqval=acqval, samples=samples)

    def _sample_forward(self, obj: Tensor) -> Tensor:
        return self.acqf._sample_forward(obj)
    
class ExpectedHypervolumeImprovementWithCost(MultiObjectiveAnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        ehvi: ExpectedHypervolumeImprovement,
        cost_model: ingredient_cost
    ):  
        super().__init__(model=model)
        self.ehvi = ehvi
        self.cost_model = cost_model

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self.ehvi.forward(X) - self.cost_model(X).squeeze(-1)

class qExpectedHypervolumeImprovementWithCost(MultiObjectiveMCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        qehvi: qExpectedHypervolumeImprovement,
        cost_model: ingredient_cost,
    ):
        super().__init__(
            model=model,
        )
        self.qehvi = qehvi
        self.cost_model = cost_model

    @t_batch_mode_transform()
    @concatenate_pending_points
    def forward(self, X: Tensor) -> Tensor:
        base_qehvi = self.qehvi.forward(X)
        return base_qehvi - self.cost_model(X).squeeze(-1)