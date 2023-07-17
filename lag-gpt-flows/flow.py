# MIT License

# Copyright (c) 2019 Chin-Wei Huang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from: https://github.com/CW-Huang/torchkit/blob/master/torchkit/flows.py

import math
import torch
from torch import nn

# To be compared to the float32 machine epsilon of 2**-23 ~= 1.2e-7
EPSILON = 1e-6



def log_sum_exp(A, dim=-1, keepdim=False):
    """
    Compute a sum in logarithm space: log(exp(a) + exp(b))
    Properly handle values which exponential cannot be represented in float32.
    """
    max_A = A.max(axis=dim, keepdim=True).values
    norm_A = A - max_A
    result = torch.log(torch.exp(norm_A).sum(axis=dim, keepdim=True)) + max_A
    if keepdim:
        return result
    else:
        return torch.squeeze(result, dim=dim)


def log_sigmoid(x):
    """
    Logarithm of the sigmoid function.
    Substract the epsilon to avoid 0 as a possible value for large x.
    """
    return -nn.functional.softplus(-x) - EPSILON

def act_a(x): return nn.functional.softplus(x) + EPSILON
def act_b(x): return x
def act_w(x): return nn.functional.softmax(x, dim=-1)

class SigmoidFlow(nn.Module):
    # Indices:
    # b: batch
    # s: sample
    # o: output dimension
    # h: hidden dimension
    # i: input dimension

    def __init__(self, hidden_dim: int, no_logit: bool = False):
        """
        A single layer of the Deep Sigmoid Flow network.
        Does not contains its parameters, they must be sent in the forward method.

        Parameters:
        -----------
        hidden_dim: uint
            The number of hidden units
        no_logit: bool, default to False
            If set to True, then the network will return a value in the (0, 1) interval.
            If kept to False, then the network will apply a logit function to this value to return
            a value in the (-inf, inf) interval.
        """
        super(SigmoidFlow, self).__init__()
        self.hidden_dim = hidden_dim
        self.no_logit = no_logit

        self.act_a = act_a
        self.act_b = act_b
        self.act_w = act_w

    def forward(self, params, x, logdet):
        """
        Transform the given value according to the given parameters,
        computing the derivative of the transformation at the same time.
        params third dimension must be equal to 3 times the number of hidden units.
        """
        # Indices:
        # b: batch
        # v: variables
        # h: hidden dimension

        # params: b, v, 3*h
        # x: b, v
        # logdet: b
        # output x: b, v
        # output logdet: b
        assert params.shape[-1] == 3 * self.hidden_dim

        a = self.act_a(params[..., : self.hidden_dim])  # b, v, h
        b = self.act_b(params[..., self.hidden_dim : 2 * self.hidden_dim])  # b, v, h
        pre_w = params[..., 2 * self.hidden_dim :]  # b, v, h
        w = self.act_w(pre_w)  # b, v, h

        pre_sigm = a * x[..., None] + b  # b, v, h
        sigm = torch.sigmoid(pre_sigm)  # b, v, h
        x_pre = (w * sigm).sum(dim=-1)  # b, v

        logj = (
            nn.functional.log_softmax(pre_w, dim=-1) + log_sigmoid(pre_sigm) + log_sigmoid(-pre_sigm) + torch.log(a)
        )  # b, v, h

        logj = log_sum_exp(logj, dim=-1, keepdim=False)  # b, v

        if self.no_logit:
            # Only keep the batch dimension, summing all others in case this method is called with more dimensions
            # Adding the passed logdet here to accumulate 
            logdet = logj.sum(dim=tuple(range(1, logj.dim()))) + logdet
            return x_pre, logdet

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5  # b, v
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)  # b, v

        logdet_ = logj + math.log(1 - EPSILON) - (torch.log(x_pre_clipped) + torch.log(-x_pre_clipped + 1))  # b, v

        # Only keep the batch dimension, summing all others in case this method is called with more dimensions
        logdet = logdet_.sum(dim=tuple(range(1, logdet_.dim()))) + logdet

        return xnew, logdet

    def forward_no_logdet(self, params, x):
        """
        Transform the given value according to the given parameters,
        but does not compute the derivative of the transformation.
        params third dimension must be equal to 3 times the number of hidden units.
        """
        # Indices: (when used for inversion)
        # b: batch
        # s: samples
        # h: hidden dimension

        # params: b, s, 3*h
        # x: b, s
        # output x: b, s
        assert params.shape[-1] == 3 * self.hidden_dim

        a = self.act_a(params[..., : self.hidden_dim])  # b, s, h
        b = self.act_b(params[..., self.hidden_dim : 2 * self.hidden_dim])  # b, s, h
        pre_w = params[..., 2 * self.hidden_dim :]  # b, s, h
        w = self.act_w(pre_w)  # b, s, h

        pre_sigm = a * x[..., None] + b  # b, s, h
        sigm = torch.sigmoid(pre_sigm)  # b, s, h
        x_pre = (w * sigm).sum(dim=-1)  # b, s

        if self.no_logit:
            return x_pre

        x_pre_clipped = x_pre * (1 - EPSILON) + EPSILON * 0.5  # b, s
        xnew = torch.log(x_pre_clipped) - torch.log(1 - x_pre_clipped)  # b, s

        return xnew


class DeepSigmoidFlow(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int):
        """
        A Deep Sigmoid Flow network, made of multiple Sigmoig Flow layers.
        This model a flexible transformation from any real values into the (0, 1) interval.
        Does not contains its parameters, they must be sent in the forward method.

        Parameters:
        -----------
        n_layers: uint
            The number of sigmoid flow layers
        hidden_dim: uint
            The number of hidden units
        """
        super(DeepSigmoidFlow, self).__init__()

        self.params_length = 3 * hidden_dim

        elayers = nn.ModuleList([SigmoidFlow(hidden_dim) for _ in range(n_layers - 1)])
        elayers += [SigmoidFlow(hidden_dim, no_logit=True)]
        self.layers = nn.Sequential(*elayers)

    @property
    def total_params_length(self):
        return len(self.layers) * self.params_length

    def forward(self, params, x):
        """
        Transform the given value according to the given parameters,
        computing the derivative of the transformation at the same time.
        params third dimension must be equal to total_params_length.
        """
        # params: batches, variables, params dim
        # x: batches, variables
        logdet = torch.zeros(
            x.shape[0],
        ).to(x.device)
        for i, layer in enumerate(self.layers):
            x, logdet = layer(params[..., i * self.params_length : (i + 1) * self.params_length], x, logdet)
        return x, logdet

    def forward_no_logdet(self, params, x):
        """
        Transform the given value according to the given parameters,
        but does not compute the derivative of the transformation.
        params third dimension must be equal to total_params_length.
        """
        # params: batches, samples, params dim
        # x: batches, samples
        for i, layer in enumerate(self.layers):
            x = layer.forward_no_logdet(params[..., i * self.params_length : (i + 1) * self.params_length], x)
        return x

    def inverse(
        self,
        marginal_params: torch.Tensor,
        u: torch.Tensor,
        max_iter: int = 100,
        precision: float = 1e-6,
        max_value: float = 1000.0,
    ) -> torch.Tensor:
        """

        NOTE: Added for compatability with Alex' notebook "ground_truth_copula.ipynb" demonstrating the 2-dimensional example
        This function actually belongs to the "marginal" class

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
        # If u has both a variable and a sample dimension, then add a singleton dimension to marginal_params to have the correct shape
        if marginal_params.dim() == u.dim():
            marginal_params = marginal_params[:, :, None, :]

        left = -max_value * torch.ones_like(u)
        right = max_value * torch.ones_like(u)
        for _ in range(max_iter):
            mid = (left + right) / 2
            error = self.forward_no_logdet(marginal_params, mid) - u
            left[error <= 0] = mid[error <= 0]
            right[error >= 0] = mid[error >= 0]

            max_error = error.abs().max().item()
            if max_error < precision:
                break
        return mid
