"""
Copyright 2022 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

>> The marginal distribution for the forecasts
"""


import torch
from torch import nn
from typing import Tuple
from flow import DeepSigmoidFlow

class DSFMarginal(nn.Module):
    """
    Compute the marginals using a Deep Sigmoid Flow conditioned using a MLP.
    The conditioning MLP uses the embedding from the encoder as its input.
    """

    def __init__(self, context_dim: int, mlp_layers: int, mlp_dim: int, flow_layers: int, flow_hid_dim: int):
        """
        Parameters:
        -----------
        context_dim: int
            Size of the context (embedding created by the encoder) that will be sent to the conditioner.
        mlp_layers: int
            Number of layers for the conditioner MLP.
        mlp_dim: int
            Dimension of the hidden layers of the conditioner MLP.
        flow_layers: int
            Number of layers for the Dense Sigmoid Flow.
        flow_hid_dim: int
            Dimension of the hidden layers of the Dense Sigmoid Flow.
        """
        super().__init__()

        self.context_dim = context_dim
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.flow_layers = flow_layers
        self.flow_hid_dim = flow_hid_dim

        self.marginal_flow = DeepSigmoidFlow(n_layers=self.flow_layers, hidden_dim=self.flow_hid_dim)

        elayers = [nn.Linear(self.context_dim, self.mlp_dim), nn.ReLU()]
        for _ in range(1, self.mlp_layers):
            elayers += [nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU()]
        elayers += [nn.Linear(self.mlp_dim, self.marginal_flow.total_params_length)]
        self.marginal_conditioner = nn.Sequential(*elayers)

    def forward_logdet(self, context: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.
        Also returns the logarithm of the derivative of this transformation.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        logdet: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The logarithm of the derivative of the transformation.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        marginal_params = self.marginal_conditioner(context)
        # If x has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]

        return self.marginal_flow.forward(marginal_params, x)

    def forward_no_logdet(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative density function of a marginal conditioned using the given context, for the given value of x.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        x: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        Returns:
        --------
        u: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The CDF at the given point, a value between 0 and 1.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of x.
        """
        marginal_params = self.marginal_conditioner(context) # [batch, series*timesteps, self.marginal_flow.total_params_length]
        # If x has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == x.dim():
            marginal_params = marginal_params[:, :, None, :]

        return self.marginal_flow.forward_no_logdet(marginal_params, x)

    def inverse(
        self,
        context: torch.Tensor,
        u: torch.Tensor,
        max_iter: int = 100,
        precision: float = 1e-6,
        max_value: float = 1000.0,
    ) -> torch.Tensor:
        """
        Compute the inverse cumulative density function of a marginal conditioned using the given context, for the given value of u.
        This method uses a dichotomic search.
        The gradient of this method cannot be computed, so it should only be used for sampling.

        Parameters:
        -----------
        context: Tensor [batch, series * time steps, embedding dimension]
            A tensor containing an embedding for each variable and time step.
            The series and time steps dimensions are merged.
        u: Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            A tensor containing the value to be transformed using the inverse CDF.
            The series and time steps dimensions are merged.
            If a third dimension is present, then the context is considered to be constant across this dimension.
        max_iter: int, default = 100
            The maximum number of iterations for the dichotomic search.
            The precision of the result should improve by a factor of 2 at each iteration.
        precision: float, default = 1e-6
            If the difference between CDF(x) and u is less than this value for all variables, stop the search.
        max_value: float, default = 1000.0
            The absolute upper bound on the possible output.
        Returns:
        --------
        x: torch.Tensor [batch, series * time steps] or [batch, series * time steps, samples]
            The inverse CDF at the given value.
            The series and time steps dimensions are merged.
            The shape of the output is the same as the shape of u.
        """
        marginal_params = self.marginal_conditioner(context)
        # If u has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == u.dim():
            marginal_params = marginal_params[:, :, None, :]

        left = -max_value * torch.ones_like(u)
        right = max_value * torch.ones_like(u)
        for _ in range(max_iter):
            mid = (left + right) / 2
            error = self.marginal_flow.forward_no_logdet(marginal_params, mid) - u
            left[error <= 0] = mid[error <= 0]
            right[error >= 0] = mid[error >= 0]

            max_error = error.abs().max().item()
            if max_error < precision:
                break
        return mid