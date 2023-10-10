from abc import ABC
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_geometric.nn as pyg_nn
from inspect import signature
from loguru import logger

from ..utils import stack_hidden

######################
# Modules from https://github.com/floriangroetschla/Recurrent-GNNs-for-algorithm-learning/blob/main/model.py
# Adapted to work with edge weights

class GRUConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr):
        super(GRUConv, self).__init__(aggr=aggr)
        logger.info(f"GRUConv: in_channels: {in_channels}, out_channels: {out_channels}")
        self.rnn = torch.nn.GRUCell(in_channels, out_channels)
        self.edge_weight_scaler = nn.Linear(1, in_channels)

    def forward(self, x, edge_index, edge_weight, last_hidden):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.rnn(out, last_hidden)
        return out

    def message(self, x_j, edge_weight):
        return F.relu(x_j + self.edge_weight_scaler(edge_weight.unsqueeze(-1)))

class GRUMLPConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, mlp_edge, aggr):
        super(GRUMLPConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(in_channels, out_channels)
        self.mlp_edge = mlp_edge

    def forward(self, x, edge_index, last_hidden, edge_weight=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = self.rnn(out, last_hidden)
        return out

    def message(self, x_j, x_i, edge_weight=None):
        concatted = torch.cat((x_j, x_i), dim=-1)
        if edge_weight is not None:
            concatted = torch.cat((concatted, edge_weight.unsqueeze(-1)), dim=-1)
        return self.mlp_edge(concatted) 

def _gruconv_module(in_channels, out_channels, aggr="add"):
    return GRUConv(in_channels, out_channels, aggr)

def _grumlpconv_module(in_channels, out_channels, aggr="add", layers=2, dropout=0.0, use_bn=False, is_weighted=False):
    input_dim = in_channels*2+1 if is_weighted else in_channels*2
    mlp  = nn.Sequential(
        nn.Linear(input_dim, in_channels)
    )
    if use_bn:
        logger.debug(f"Using batch norm in GIN module")
        mlp.add_module(f"bn_input", nn.BatchNorm1d(in_channels))
    for _ in range(layers-1):
        mlp.add_module(f"relu_{_}", nn.ReLU())
        mlp.add_module(f"linear_{_}", nn.Linear(in_channels, in_channels))
        if use_bn:
            logger.debug(f"Using batch norm in GIN module")
            mlp.add_module(f"bn_{_}", nn.BatchNorm1d(in_channels))
    if dropout > 0:
        mlp.add_module(f"dropout", nn.Dropout(dropout))
    return GRUMLPConv(in_channels, out_channels, mlp, aggr)

######################

def _gin_module(in_channels, out_channels, eps=0, train_eps=False, layers=2, dropout=0.0, use_bn=False, aggr="add"):
    mlp = nn.Sequential(
        nn.Linear(in_channels, out_channels),
    )
    if use_bn:
        logger.debug(f"Using batch norm in GIN module")
        mlp.add_module(f"bn_input", nn.BatchNorm1d(out_channels))
    for _ in range(layers-1):
        mlp.add_module(f"relu_{_}", nn.ReLU())
        mlp.add_module(f"linear_{_}", nn.Linear(out_channels, out_channels))
        if use_bn:
            logger.debug(f"Using batch norm in GIN module")
            mlp.add_module(f"bn_{_}", nn.BatchNorm1d(out_channels))
    if dropout > 0:
        mlp.add_module(f"dropout", nn.Dropout(dropout))
    return pyg_nn.GINConv(mlp, eps, train_eps, aggr=aggr)

def _gine_module(in_channels, out_channels, eps=0, train_eps=False, layers=2, dropout=0.0, use_bn=False, edge_dim=1, aggr="add"):
    
    mlp = nn.Sequential(
        nn.Linear(in_channels, out_channels),
    )
    if use_bn:
        logger.debug(f"Using batch norm in GIN module")
        mlp.add_module(f"bn_input", nn.BatchNorm1d(out_channels))
    for _ in range(layers-1):
        mlp.add_module(f"relu_{_}", nn.ReLU())
        mlp.add_module(f"linear_{_}", nn.Linear(out_channels, out_channels))
        if use_bn:
            logger.debug(f"Using batch norm in GIN module")
            mlp.add_module(f"bn_{_}", nn.BatchNorm1d(out_channels))
    if dropout > 0:
        mlp.add_module(f"dropout", nn.Dropout(dropout))
    
    return pyg_nn.GINEConv(mlp, eps, train_eps, edge_dim=edge_dim, aggr=aggr)

def _get_processor(name):
    if name == "GCNConv":
        return pyg_nn.GCNConv
    elif name == "GINConv":
        return _gin_module    
    elif name == "GINEConv":
        return _gine_module
    elif name == "GRUConv":
        return _gruconv_module
    elif name == "RecGNNConv": # initially called GRUMLPConv
        return _grumlpconv_module
    else:
        raise ValueError(f"Unknown processor {name}")
    
class Processor(nn.Module, ABC):
    def __init__(self, cfg, has_randomness=False):
        super().__init__()
        self.cfg = cfg        
        processor_input = self.cfg.MODEL.HIDDEN_DIM*3 if self.cfg.MODEL.PROCESSOR_USE_LAST_HIDDEN else self.cfg.MODEL.HIDDEN_DIM*2
        if has_randomness:
            processor_input += 1
        self.core = _get_processor(self.cfg.MODEL.PROCESSOR.NAME)(in_channels=processor_input, out_channels=self.cfg.MODEL.HIDDEN_DIM, **self.cfg.MODEL.PROCESSOR.KWARGS[0])
        if self.cfg.MODEL.PROCESSOR.LAYERNORM.ENABLE:
            self.norm = pyg_nn.LayerNorm(self.cfg.MODEL.HIDDEN_DIM, mode=self.cfg.MODEL.PROCESSOR.LAYERNORM.MODE)
        
        self._core_requires_last_hidden = "last_hidden" in signature(self.core.forward).parameters

    def forward(self, input_hidden, hidden, last_hidden, batch_assignment, randomness=None, **kwargs):
        stacked = stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.PROCESSOR_USE_LAST_HIDDEN)
        if randomness is not None:
            stacked = torch.cat((stacked, randomness.unsqueeze(1)), dim=-1)
        if self._core_requires_last_hidden:
            kwargs["last_hidden"] = last_hidden
        out = self.core(stacked, **kwargs)
        if self.cfg.MODEL.PROCESSOR.LAYERNORM.ENABLE:
            # norm
            out = self.norm(out, batch=batch_assignment)
        return out

    def has_edge_weight(self):
        return "edge_weight" in signature(self.core.forward).parameters
    
    def has_edge_attr(self):
        return "edge_attr" in signature(self.core.forward).parameters



