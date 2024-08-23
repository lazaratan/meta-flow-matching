import sys
from pathlib import Path
# Add the directory one level up to sys.path
current_directory = Path(__file__).absolute().parent
parent_directory = current_directory.parent
sys.path.append(str(parent_directory))


import pandas as pd
import numpy as np
import torch
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
from dataclasses import dataclass


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x)


class torch_wrapper_flow_cond(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        padding = torch.zeros((x.shape[0], self.model.num_conditions)).long()
        out = self.model(t, x)
        return torch.cat([out, padding.to(x)], dim=1)
    

class torch_wrapper_gnn_flow_cond(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        padding = torch.zeros((x.shape[0], self.model.num_treat_conditions)).long()
        out = self.model(t, x)
        return torch.cat([out, padding.to(x)], dim=1)


class torch_wrapper_tv(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs): # 
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x)

class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class MLP_cond(torch.nn.Module):
    """Class conditional, assume the last dimension(s) are cond"""
    def __init__(self, dim, cond=1, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.dim = dim
        self.conditional = int(cond)
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0) + int(cond), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x, *args, **kwargs):
        out = self.net(x)
        # cat class conditional
        return torch.cat([out, x[:, -int(self.conditional):]], 1)
   
    
def mlp(input_size, output_size, hidden_size, num_layers, act_fn=torch.nn.SELU): #act_fn=torch.nn.SiLU
    layers = []
    prev_size = input_size
    for _ in range(num_layers-1):
        layers.append(torch.nn.Linear(prev_size, hidden_size))
        layers.append(act_fn())
        prev_size = hidden_size
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)


class SkipMLP(torch.nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size, num_layers, act_fn=torch.nn.SELU
    ):
        super(SkipMLP, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.act_fn = act_fn

        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.activation = act_fn()

    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out) + out)  # Skip connection
        out = self.output_layer(out)
        return out

@dataclass(eq=False)
class Flow(torch.nn.Module):
    def __init__(
        self,
        D: int = 2,
        num_temporal_freqs: int = 3,
        num_spatial_samples: int = 128,
        spatial_feat_scale: float = 0.01,
        num_hidden: int = 512,
        num_layers: int = 4,
        num_conditions: int = None,
        num_treat_conditions: int = 0,
        t_embedding_dim: int = 128,
        skip_connections: bool = True,
    ) -> None:
        super().__init__()
        self.D = D
        self.num_temporal_freqs = num_temporal_freqs
        self.num_spatial_samples = num_spatial_samples
        self.spatial_feat_scale = spatial_feat_scale
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_conditions = (
            num_conditions + num_treat_conditions
            if num_conditions is not None
            else num_treat_conditions
        )
        self.t_embedding_dim = t_embedding_dim
        self.skip_connections = skip_connections

        if self.num_conditions is not None:
            input_size = (
                self.num_conditions
                + 2 * self.num_spatial_samples
                + t_embedding_dim
                + self.D
            )

            if self.skip_connections:
                self.net = SkipMLP(input_size, self.D, self.num_hidden, self.num_layers)
            else:
                self.net = mlp(input_size, self.D, self.num_hidden, self.num_layers)
            
        else:
            input_size = (
                2 * self.num_spatial_samples
                + t_embedding_dim
                + self.D
            )
            #self.net = mlp(input_size, self.D, self.num_hidden, self.num_layers)
            self.net = SkipMLP(input_size, self.D, self.num_hidden, self.num_layers)

        self.temporal_freqs = (
            torch.arange(1, self.num_temporal_freqs + 1, device="cuda") * torch.pi
        )
        
        self.B = (
            torch.randn((self.D, self.num_spatial_samples), device="cuda")
            * self.spatial_feat_scale
        )
            
    def get_timestep_embedding(self, t, max_positions=10000):
        assert len(t.shape) == 1 
        half_dim = self.t_embedding_dim // 2
        # magic number 10000 is from transformers
        emb = torch.tensor(math.log(max_positions) / (half_dim - 1))
        emb = torch.exp(torch.arange(half_dim) * -emb).to(t)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
        if self.t_embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, [[0, 0], [0, 1]])
        assert emb.shape == (t.shape[0], self.t_embedding_dim)
        return emb

    def forward(self, t, x):
        assert x.ndim == 2
        
        if len(t.shape) != 1:
            t = t.unsqueeze(0)
        t = self.get_timestep_embedding(t)
        t = t.expand(*x.shape[:-1], -1)
 
        orig_x = x
        x = 2.0 * torch.pi * x[:, :self.D] @ self.B
        x = torch.cat((x.cos(), x.sin()), dim=-1)

        return self.net(torch.cat((t, x, orig_x), dim=-1))