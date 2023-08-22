import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
from loguru import logger
from sklearn.metrics import f1_score

from .models import EncodeProcessDecode
from .loss import CLRSLoss
from .utils import stack_dicts
from .metrics import calc_metrics

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CLRSModel(pl.LightningModule):
    def __init__(self, specs, cfg):
        super().__init__()
        self.hparams.update(cfg)
        self.cfg = cfg
        self.model = EncodeProcessDecode(specs, cfg)
        self.loss = CLRSLoss(specs, cfg.TRAIN.LOSS.HIDDEN_LOSS_TYPE)
        self.step_output_cache = defaultdict(list)
        self.current_loader_idx = 0
        self.specs = specs
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)        

    def _loss(self, batch, output, hints, hidden):
        outloss, hintloss, hiddenloss = self.loss(batch, output, hints, hidden)
        loss = self.cfg.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT * outloss + self.cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT * hintloss + self.cfg.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT * hiddenloss
        return loss, outloss, hintloss, hiddenloss

    def training_step(self, batch, batch_idx):
        output, hints, hidden = self.model(batch)
        loss, outloss, hintloss, hiddenloss = self._loss(batch, output, hints, hidden)
        self.log("train/outloss", outloss, batch_size=batch.num_graphs)
        self.log("train/hintloss", hintloss, batch_size=batch.num_graphs)
        self.log("train/hiddenloss", hiddenloss, batch_size=batch.num_graphs)
        self.log("train/loss", loss, batch_size=batch.num_graphs)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def _shared_eval(self, batch, dataloader_idx, stage):
        output, hints, hidden = self.model(batch)
        loss, outloss, hintloss, hiddenloss = self._loss(batch, output, hints, hidden)
        self.current_loader_idx = dataloader_idx
        # calc batch metrics
        assert len(batch.outputs) == 1
        metrics = calc_metrics(batch.outputs[0], output, batch, self.specs[batch.outputs[0]][2])
        output.update({f"{m}_metric": metrics[m] for m in metrics})
        output["batch_size"] = torch.tensor(batch.num_graphs).float()
        output["num_nodes"] = torch.tensor(batch.num_nodes).float()
        return loss, output

    def _end_of_epoch_metrics(self, dataloader_idx):
        output = stack_dicts(self.step_output_cache[dataloader_idx])
        # average metrics over graphs
        metrics = {}
        for m in output:
            if not m.endswith("_metric"):
                continue
            if m.startswith("graph"):
                # graph level metrics have to be computed differently
                metrics["graph_accuracy"] = output[m].float().mean()
                metrics["graph_f1"] = f1_score(torch.ones_like(output[m]).cpu().numpy(), output[m].cpu().numpy(), average='binary')
            else:
                metrics[m[:-7]] = output[m].float().mean()
        return metrics
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, output = self._shared_eval(batch, dataloader_idx, "val")
        self.log(f'val/loss/{self.trainer.datamodule.get_val_loader_nickname(dataloader_idx)}', loss, batch_size=batch.num_graphs, add_dataloader_idx=False)

        self.step_output_cache[dataloader_idx].append(output)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, output = self._shared_eval(batch, dataloader_idx, "test")
        self.log(f'test/loss/{self.trainer.datamodule.get_test_loader_nickname(dataloader_idx)}', loss, batch_size=batch.num_graphs, add_dataloader_idx=False)

        self.step_output_cache[dataloader_idx].append(output)
        return loss
    
    def on_validation_epoch_end(self):
        for dataloader_idx in self.step_output_cache.keys():
            metrics = self._end_of_epoch_metrics(dataloader_idx)
            for key in metrics:
                self.log(f"val/{key}", metrics[key], add_dataloader_idx=False)
        self.step_output_cache.clear()

    def on_test_epoch_end(self):
        for dataloader_idx in self.step_output_cache.keys():
            metrics = self._end_of_epoch_metrics(dataloader_idx)
            for key in metrics:
                self.log(f"test/{key}/{self.trainer.datamodule.get_test_loader_nickname(dataloader_idx)}", metrics[key], add_dataloader_idx=False)
        self.step_output_cache.clear()  

    def configure_optimizers(self):
        if self.cfg.TRAIN.OPTIMIZER.NAME == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.TRAIN.OPTIMIZER.LR)
        elif self.cfg.TRAIN.OPTIMIZER.NAME == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.OPTIMIZER.LR)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.TRAIN.OPTIMIZER.NAME} not implemented")
        out = {"optimizer": optimizer, "monitor": "val/loss/0", "interval": "step", "frequency": 1}
        if self.cfg.TRAIN.SCHEDULER.ENABLE:
            try:
                scheduler = getattr(torch.optim.lr_scheduler, self.cfg.TRAIN.SCHEDULER.NAME)(optimizer, **self.cfg.TRAIN.SCHEDULER.PARAMS[0])
                out["lr_scheduler"] = scheduler
                out['monitor'] = 'val/loss/0'
                
            except AttributeError:
                raise NotImplementedError(f"Scheduler {self.cfg.TRAIN.SCHEDULER.NAME} not implemented")

        return out

