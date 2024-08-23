"""
Adopted from https://github.com/bunnech/cellot/blob/main/cellot/networks/icnns.py
"""

import numpy as np
import torch
from torch import autograd

from torch import nn
from numpy.testing import assert_allclose

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}

class NonNegativeLinear(nn.Linear):
    def __init__(self, *args, beta=1.0, **kwargs):
        super(NonNegativeLinear, self).__init__(*args, **kwargs)
        self.beta = beta
        return

    def forward(self, x):
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        return nn.functional.softplus(self.weight, beta=self.beta)


class ICNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units,
        activation="LeakyReLU",
        softplus_W_kernels=False,
        softplus_beta=1,
        std=0.1,
        fnorm_penalty=0,
        kernel_init_fxn=None,
    ):
        super(ICNN, self).__init__()
        self.fnorm_penalty = fnorm_penalty
        self.softplus_W_kernels = softplus_W_kernels

        if isinstance(activation, str):
            activation = ACTIVATIONS[activation.lower().replace("_", "")]
        self.sigma = activation

        units = hidden_units + [1]

        if self.softplus_W_kernels:

            def WLinear(*args, **kwargs):
                return NonNegativeLinear(*args, **kwargs, beta=softplus_beta)

        else:
            WLinear = nn.Linear

        self.W = nn.ModuleList(
            [
                WLinear(idim, odim, bias=False)
                for idim, odim in zip(units[:-1], units[1:])
            ]
        )

        self.A = nn.ModuleList(
            [nn.Linear(input_dim, odim, bias=True) for odim in units]
        )

        if kernel_init_fxn is not None:
            for layer in self.A:
                kernel_init_fxn(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in self.W:
                kernel_init_fxn(layer.weight)

        return

    def forward(self, x):
        x = x.float().to(self.W[0].weight.device).requires_grad_(True)
        z = self.sigma(0.2)(self.A[0](x)).requires_grad_(True)
        z = z * z
        z = z.requires_grad_(True)

        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(0.2)(W(z) + A(x))

        y = self.W[-1](z) + self.A[-1](x)
        return y

    @torch.inference_mode(False)
    def transport(self, x):
        assert x.requires_grad
        x = x.clone().detach().requires_grad_(True)
        x = x.float().cpu().requires_grad_(True)
        (output,) = autograd.grad(
            self.forward(x),
            x.cpu(),
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device="cpu").float(),
        )
        return output

    def clamp_w(self):
        if self.softplus_W_kernels:
            return

        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0)
        return

    def penalize_w(self):
        return self.fnorm_penalty * sum(
            map(lambda x: torch.nn.functional.relu(-x.weight).norm(), self.W)
        )


def test_icnn_convexity(icnn):
    data_dim = icnn.A[0].in_features

    zeros = np.zeros(100)
    for _ in range(100):
        x = torch.rand((100, data_dim))
        y = torch.rand((100, data_dim))

        fx = icnn(x)
        fy = icnn(y)

        for t in np.linspace(0, 1, 10):
            fxy = icnn(t * x + (1 - t) * y)
            res = (t * fx + (1 - t) * fy) - fxy
            res = res.detach().numpy().squeeze()
            assert_allclose(np.minimum(res, 0), zeros, atol=1e-6)
