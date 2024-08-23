import torch
import math
from torch import nn
from dataclasses import dataclass

import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

from src.models.components.mlp import mlp, SkipMLP    
    
@dataclass(eq=False)
class GlobalGNN(nn.Module):
    D: int = 2
    num_temporal_freqs: int = 3
    num_spatial_samples: int = 128
    spatial_feat_scale: float = 0.01
    t_embedding_dim: int = 128
    num_hidden_gnn: int = 512
    num_layers_gnn: int = 3
    num_hidden_decoder: int = 512
    num_layers_decoder: int = 4
    knn_k: int = 100
    num_treat_conditions: int = None
    num_cell_conditions: int = None
    skip_connections: bool = True

    def __post_init__(self):
        super().__init__()

        # init GNN
        self.gcn_convs = nn.ModuleList()
        if self.num_cell_conditions is None:
            self.gcn_convs.append(GCNConv(self.D, self.num_hidden_gnn))
        else:
            self.gcn_convs.append(
                GCNConv(self.D + self.num_cell_conditions, self.num_hidden_gnn)
            )
        for _ in range(self.num_layers_gnn - 1):
            self.gcn_convs.append(GCNConv(self.num_hidden_gnn, self.num_hidden_gnn))

        # init Flow
        if self.num_treat_conditions is None:
            input_size = (
                self.num_hidden_gnn
                + self.t_embedding_dim
                + 2 * self.num_spatial_samples
                + self.D
            )
            self.decoder = SkipMLP(
                input_size, self.D, self.num_hidden_decoder, self.num_layers_decoder
            )

            self.temporal_freqs = (
                torch.arange(1, self.num_temporal_freqs + 1, device="cuda") * torch.pi
            )
        else:
            input_size = (
                self.num_hidden_gnn
                + self.t_embedding_dim
                + 2 * self.num_spatial_samples
                + self.D
                + self.num_treat_conditions
            )

            if self.skip_connections:
                self.decoder = SkipMLP(
                    input_size, self.D, self.num_hidden_decoder, self.num_layers_decoder
                )
            else:
                self.decoder = mlp(
                    input_size, self.D, self.num_hidden_decoder, self.num_layers_decoder
                )

            self.temporal_freqs = (
                torch.arange(1, self.num_temporal_freqs + 1, device="cuda") * torch.pi
            )
            
        self.B = (
            torch.randn((self.D, self.num_spatial_samples), device="cuda")
            * self.spatial_feat_scale
        )
    
    def embed_source(self, source_samples, cond=None):        
        if len(source_samples.shape) > 2:
            b, n, d = source_samples.shape
            data_list = []

            for i in range(b):
                x = source_samples[i]
                edge_index = torch_geometric.nn.pool.knn_graph(x.cpu(), k=self.knn_k)
                
                if cond is not None:
                    cond_i = cond[i] if len(cond.shape) > 2 else cond
                    x = torch.cat([x, cond_i], dim=-1)
                    
                data_list.append(
                    torch_geometric.data.Data(x=x.cuda(), edge_index=edge_index.cuda())
                )
            
            # Create a Batch object from the list of Data objects
            batch_data = torch_geometric.data.Batch.from_data_list(data_list)
            z = batch_data.x
            edge_index = batch_data.edge_index
        else:
            edge_index = torch_geometric.nn.pool.knn_graph(
                x=source_samples.cpu(), k=self.knn_k
            ).cuda()
            z = source_samples
            
        if cond is not None and len(source_samples.shape) == 2:
            z = self.gcn_convs[0](torch.cat((z, cond), dim=-1), edge_index)
            for conv in self.gcn_convs[1:-1]:
                z = conv(z, edge_index)
                z = F.relu(z)
        else:
            for conv in self.gcn_convs[:-1]:
                z = conv(z, edge_index)
                z = F.relu(z)

        z = self.gcn_convs[-1](z, edge_index)
        
        if len(source_samples.shape) > 2:
            z = z.view(b, n, -1).mean(dim=1)
            z_norm = torch.norm(z, dim=1, keepdim=True)
        else:
            z = z.mean(dim=0, keepdim=True)
            z_norm = torch.norm(z)
        
        z = z / z_norm
        return z
    
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

    def flow(self, embedding, t, y):
        assert y.ndim == 2

        if len(t.shape) != 1:
            t = t.unsqueeze(0)
        t = self.get_timestep_embedding(t)
        t = t.expand(*y.shape[:-1], -1)

        orig_y = y
        y = 2.0 * torch.pi * y[:, :self.D] @ self.B
        y = torch.cat((y.cos(), y.sin()), dim=-1)

        embedding = embedding.expand(y.shape[0], -1)

        z = torch.cat((embedding, t, y, orig_y), dim=-1)

        return self.decoder(z)

    def update_embedding_for_inference(self, source_samples, cond=None):
        self.embedding = self.embed_source(source_samples, cond=cond).detach()

    def forward(self, t, x):
        return self.flow(self.embedding, t, x)