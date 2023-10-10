import torch.nn as nn
import torch
import torch_scatter
from loguru import logger
from torch_geometric.nn import global_mean_pool, global_max_pool
from ..utils import NaNException

## Node decoders

class NodeBaseDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, *args, **kwargs):
        x = self.lin(x)
        return x

class NodeScalarDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, *args, **kwargs):
        out = super().forward(x).squeeze(-1)
        return out

class NodeMaskDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, *args, **kwargs):
        out = super().forward(x).squeeze(-1)
        return out

class NodeMaskOneDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x) # N x 1

        out = torch_scatter.scatter_log_softmax(out, batch_assignment, dim=0)
        return out


class NodeCategoricalDecoder(NodeBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x) # N x C
        out = torch.log_softmax(out, dim=-1)
        return out



#### Edge decoders

class BaseEdgeDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.source_lin = nn.Linear(hidden_dim, hidden_dim)
        self.target_lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hiddens, edge_index):
        zs = self.source_lin(hiddens) # N x H
        zt = self.target_lin(hiddens) # N x H
        return (zs[edge_index[0]] * zt[edge_index[1]]).sum(dim=-1)
    
class EdgeMaskDecoder(BaseEdgeDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, hiddens, edge_index, **kwargs):
        out = super().forward(hiddens, edge_index).sigmoid().squeeze(-1)
        return out
    
class NodePointerDecoder(BaseEdgeDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, hiddens, edge_index, **kwargs):
        z =  super().forward(hiddens, edge_index) # E
        # per node outgoing softmax
        z = torch_scatter.scatter_log_softmax(z, edge_index[0], dim=0)
        return z

#### Graph decoders

class GraphBaseDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lin = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, batch_assignment, **kwargs):
        x = self.lin(x)
        out = global_mean_pool(x, batch_assignment)
        return out.squeeze(-1)
    
class GraphMaskDecoder(GraphBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x, batch_assignment)
        out = out.sigmoid()
        return out

class GraphCategoricalDecoder(GraphBaseDecoder):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__(input_dim, hidden_dim)

    def forward(self, x, batch_assignment, **kwargs):
        out = super().forward(x, batch_assignment)
        out = torch.log_softmax(out, dim=-1)
        return out
    

_DECODER_MAP = {
    ('node', 'scalar'): NodeScalarDecoder,
    ('node', 'mask'): NodeMaskDecoder,
    ('node', 'mask_one'): NodeMaskOneDecoder,
    ('node', 'pointer'): NodePointerDecoder,
    ('node', 'categorical'): NodeCategoricalDecoder,
    ('edge', 'mask'): EdgeMaskDecoder,
    ('edge', 'scalar'): BaseEdgeDecoder,
    ('graph', 'scalar'): GraphBaseDecoder,  
    ('graph', 'mask'): GraphMaskDecoder,
    ('graph', 'categorical'): GraphCategoricalDecoder,
}
    
class Decoder(nn.Module):
    def __init__(self, specs, hidden_dim=128, no_hint=False):
        super().__init__()
        self.specs = specs
        self.hidden_dim = hidden_dim
        self.decoder = nn.ModuleDict()
        for k, v in specs.items():
            stage, loc, type_, cat_dim = v
            if no_hint and stage == 'hint':
                logger.debug(f'Ignoring hint decoder for {k}')
                continue
            if stage == 'input':
                logger.debug(f'Ignoring input decoder for {k}')
                continue
            if stage == 'hint':
                k = k.replace('_h', '')
            
            input_dim = 1
            if type_ == 'categorical':
                input_dim = cat_dim

            if k not in self.decoder:
                self.decoder[k] = _DECODER_MAP[(loc, type_)](input_dim, hidden_dim)

    def forward(self, hidden, batch, stage):
        output = {}
        for key in getattr(batch, stage):
            if stage == 'hints':
                dkey = key.replace('_h', '')
            else:
                dkey = key

            output[key] = self.decoder[dkey](hidden, edge_index=batch.edge_index, batch_assignment=batch.batch)
        return output

    
def grab_outputs(hints, batch):
    """This function grabs the outputs from the batch and returns them in the same format as the hints"""
    output = {}
    for k in hints:
        k_out = k.replace('_h', '')
        if k_out in batch.outputs:
            output[k_out] = hints[k]
    return output

def output_mask(batch, step):
    final_node_idx = (batch.length[batch.batch]-1)

    masks = {}
    for key in batch.outputs:
        if key in batch.edge_attrs():
            final_edge_idx = final_node_idx[batch.edge_index[0]]
            masks[key] = final_edge_idx == step
        elif key in batch.node_attrs():
            masks[key] = final_node_idx == step
        else:
            # graph attribute
            masks[key] = batch.length == step + 1
    return masks