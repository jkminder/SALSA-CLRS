import torch
import torch_geometric.nn as nn
from inspect import signature
from loguru import logger

from .encoder import Encoder
from .decoder import Decoder, grab_outputs, output_mask
from .processor import Processor
from ..utils import stack_hidden   

def stack_hints(hints):
    return {k: torch.stack([hint[k] for hint in hints], dim=-1) for k in hints[0]} if hints else {}

class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, specs, cfg):
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.has_randomness = 'randomness' in specs
        self.processor = Processor(cfg, self.has_randomness)
        self.encoder = Encoder(specs, self.cfg.MODEL.HIDDEN_DIM)

        decoder_input = self.cfg.MODEL.HIDDEN_DIM*3 if self.cfg.MODEL.DECODER_USE_LAST_HIDDEN else self.cfg.MODEL.HIDDEN_DIM*2
        self.decoder = Decoder(specs, decoder_input, no_hint=self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0)
        logger.debug(f"Decoder: {self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT == 0.0}")

        if not self.processor.has_edge_weight() and not self.processor.has_edge_attr():
            if "A" in specs:
                logger.warning(f"Processor {self.cfg.MODEL.PROCESSOR.NAME} does neither support edge_weight nor edge_attr, but the algorithm requires edge weights.")
                raise ValueError(f"Processor {self.cfg.MODEL.PROCESSOR.NAME} does neither support edge_weight nor edge_attr, but the algorithm requires edge weights.")
        elif self.processor.has_edge_weight():
            self.edge_weight_name = "edge_weight"
        elif self.processor.has_edge_attr():
            self.edge_weight_name = "edge_attr"


        if self.cfg.MODEL.GRU.ENABLE:
            self.gru = torch.nn.GRUCell(self.cfg.MODEL.HIDDEN_DIM, self.cfg.MODEL.HIDDEN_DIM)
        
    def process_weights(self, batch):
        if self.edge_weight_name == "edge_attr":
            return batch.weights.unsqueeze(-1).type(torch.float32)
        else:
            return batch.weights
        
    def forward(self, batch):
        input_hidden, randomness = self.encoder(batch)
        max_len = batch.length.max().item()
        hints = []
        output = None

        # Process for length
        hidden = input_hidden
        for step in range(max_len):
            last_hidden = hidden
            for _ in range(self.cfg.MODEL.MSG_PASSING_STEPS):
                hidden = self.processor(input_hidden, hidden, last_hidden, randomness=randomness[:, step] if randomness is not None else None, edge_index=batch.edge_index, batch_assignment=batch.batch, **{self.edge_weight_name: self.process_weights(batch) for _ in range(1) if hasattr(batch, 'weights') })
                if self.cfg.MODEL.GRU.ENABLE:
                    hidden = self.gru(hidden, last_hidden)
            if self.training and self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                hints.append(self.decoder(stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'hints'))

            # Check if output needs to be constructed
            if (batch.length == step+1).sum() > 0:
                # Decode outputs
                if self.training and self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT > 0.0:
                    # The last hint is the output, no need to decode again, its the same decoder
                    output_step = grab_outputs(hints[-1], batch)
                else:
                    output_step = self.decoder(stack_hidden(input_hidden, hidden, last_hidden, self.cfg.MODEL.DECODER_USE_LAST_HIDDEN), batch, 'outputs')
                
                # Mask output
                mask = output_mask(batch, step)   
                if output is None:
                    output = {k: output_step[k]*mask[k] for k in output_step}
                else:
                    for k in output_step:
                        output[k][mask[k]] = output_step[k][mask[k]]

        hints = stack_hints(hints)

        return output, hints, hidden

