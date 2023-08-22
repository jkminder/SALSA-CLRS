import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from functools import partial
from loguru import logger
from .utils import NaNException

def calculate_loss(mask, truth, pred, edge_index, type_, batch_assignment):
    if type_ == "scalar":
        return torch.mean(F.mse_loss(pred, truth, reduction='none') * mask)
    elif type_ == "mask":
        # pred is not sigmoided due to autocast issues
        return torch.mean(F.binary_cross_entropy_with_logits(pred, truth, reduction='none') * mask)
    elif type_ == "mask_one":
        # cross entropy loss, pred is logsoftmaxed
        logits = truth*pred*mask
        return (-torch_scatter.scatter(logits, batch_assignment, dim=0)).mean()
    elif type_ == "categorical":
        # Per node cross entropy loss, pred is logsoftmaxed
        pred = pred.permute(0, 2, 1) # H x C x N -> H x N x C
        categories = pred.shape[-1]
        # repeat mask for each category
        mask = mask.unsqueeze(-1).repeat_interleave(categories, dim=-1) # H x N -> H x N x C
        logits = truth*pred*mask # (H x) N x C
        return (-torch.sum(logits, dim=-1)).mean()
    elif type_ == "pointer":
        # pred is logsoftmaxed
        logits = truth*pred*mask
        loss = (-torch_scatter.scatter(logits, edge_index[0], dim=0))
        return loss.mean()
    else:
        raise NotImplementedError
    
class CLRSLoss(torch.nn.Module):
    def __init__(self, specs, hidden_loss_type):
        super().__init__()
        self.specs = specs

        if hidden_loss_type == "l2":
            self.hidden_loss = lambda x: torch.mean(torch.linalg.norm(x, dim=1))
        else:
            raise NotImplementedError(f"Unknown hidden loss type {hidden_loss_type}")

    def forward(self, batch, outputs, hints, hidden):
        device = batch.edge_index.device
        output_loss = torch.zeros(1, device=device)
        for key in outputs:
            # check of nan
            if torch.isnan(outputs[key]).any(): 
                logger.warning(f"NaN in {key} output")
                raise NaNException(f"NaN in {key} output")
            stage, loc, type_, cat_dim = self.specs[key]
            mask = torch.ones_like(batch[key])
            output_loss += calculate_loss(mask, batch[key], outputs[key], batch.edge_index,  type_, batch.batch)

        hint_loss = torch.zeros(1, device=device)
        final_node_idx = (batch.length[batch.batch]-1)
        final_edge_idx = final_node_idx[batch.edge_index[0]]
        for key in hints:
            # check of nan
            if torch.isnan(hints[key]).any():
                logger.warning(f"NaN in {key} hint")
            stage, loc, type_, cat_dim = self.specs[key]
            if key in batch.edge_attrs():
                mask = torch.arange(batch.length.max(), device=device).unsqueeze(0) <= final_edge_idx.unsqueeze(1)
            elif key in batch.node_attrs():
                mask = torch.arange(batch.length.max(), device=device).unsqueeze(0) <= final_node_idx.unsqueeze(1)
            else:
                # graph attribute
                mask = torch.arange(batch.length.max(), device=device).unsqueeze(0) <= batch.length.unsqueeze(1)
            hint_loss += calculate_loss(mask, batch[key], hints[key], batch.edge_index, type_, batch.batch)
        
        return output_loss, hint_loss, self.hidden_loss(hidden)