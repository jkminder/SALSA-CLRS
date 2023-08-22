import logging
import os
import sys
from typing import Optional, Any, Dict
from collections import defaultdict
import csv

import torch
from loguru import logger
import lightning.pytorch as pl
import warnings
import wandb
import argparse
import os
import math

from core.module import CLRSModel
from core.config import load_cfg
from core.utils import NaNException

from salsaclrs import CLRSDataModule, load_dataset

logger.remove()
logger.add(sys.stderr, level="INFO")


def train(model, datamodule, cfg, enable_wandb, specs, seed=42, checkpoint_dir=None):
    if enable_wandb:
        wandblogger = pl.loggers.WandbLogger(project=cfg.LOGGING.WANDB.PROJECT, entity=cfg.LOGGING.WANDB.ENTITY, group=cfg.LOGGING.WANDB.GROUP, name=cfg.RUN_NAME+"-"+str(args.seed))
    else:
        wandblogger = None

    callbacks = []
    # checkpointing
    if checkpoint_dir is not None:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(cfg.DATA.ROOT, "checkpoints", str(cfg.ALGORITHM), cfg.RUN_NAME), monitor="val/loss/0", mode="min", filename=f'seed{seed}-{{epoch}}-{{step}}', save_top_k=1, verbose=True)
        callbacks.append(ckpt_cbk)

    # early stopping
    early_stop_cbk = pl.callbacks.EarlyStopping(monitor="val/loss/0", patience=cfg.TRAIN.EARLY_STOPPING_PATIENCE, mode="min", verbose=True)
    callbacks.append(early_stop_cbk)

    # Setup trainer
    trainer = pl.Trainer(
        enable_checkpointing=True,
        callbacks=[ckpt_cbk, early_stop_cbk],
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        logger=wandblogger,
        accelerator="auto",
        log_every_n_steps=5,
        gradient_clip_val=cfg.TRAIN.GRADIENT_CLIP_VAL,
        reload_dataloaders_every_n_epochs=datamodule.reload_every_n_epochs,
        precision= cfg.TRAIN.PRECISION,
    )

    # Load checkpoint
    if cfg.TRAIN.LOAD_CHECKPOINT is not None:
        logger.info(f"Loading checkpoint from {cfg.TRAIN.LOAD_CHECKPOINT}")
        model = CLRSModel.load_from_checkpoint(cfg.TRAIN.LOAD_CHECKPOINT, cfg=cfg, specs=specs)

    # Train
    if cfg.TRAIN.ENABLE:
        try:
            logger.info("Starting training...")
            trainer.fit(model, datamodule=datamodule)
        except NaNException:
            logger.info(f"NaN detected, trying to recover from {ckpt_cbk.best_model_path}...")
            try:
                trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_cbk.best_model_path)
            except NaNException:
                logger.info("Recovery failed, stopping training...")

    # Load best model
    if cfg.TRAIN.LOAD_CHECKPOINT is None and cfg.TRAIN.ENABLE:
        logger.info(f"Best model path: {ckpt_cbk.best_model_path}")
        model = CLRSModel.load_from_checkpoint(ckpt_cbk.best_model_path)

    # Test
    logger.info("Testing best model...")
    results = trainer.test(model, datamodule=datamodule)

    # Log results
    stacked_results = {}
    for d in results:
        stacked_results.update(d)

    logger.info(stacked_results)
    logger.info("Saving results...")
    results_dir = f"results/{cfg.ALGORITHM}/{cfg.RUN_NAME}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # write csv
    with open(os.path.join(results_dir, f"{seed}.csv"), "w") as f:
        writer = csv.DictWriter(f, stacked_results.keys())
        writer.writeheader()
        writer.writerow(stacked_results)



def load_ds(cfg, DATA_DIR):
    train_ds = load_dataset(cfg.ALGORITHM, "train", DATA_DIR)
    val_ds = load_dataset(cfg.ALGORITHM, "val", DATA_DIR)
    test_datasets = load_dataset(cfg.ALGORITHM, "test", DATA_DIR) # dict of datasets
    return train_ds, val_ds, list(test_datasets.values())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--enable-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hints", action="store_true", help="Use hints.")
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    # set seed
    pl.seed_everything(args.seed)
    logger.info(f"Using seed {args.seed}")

    # load config
    cfg = load_cfg(args.cfg)


    if args.hints:
        cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT = 1.0
        cfg.RUN_NAME = cfg.RUN_NAME+"-hints"
        logger.info("Using hints.")

    
    logger.info("Starting run...")
    torch.set_float32_matmul_precision('medium')

    # load datasets
    train_ds, val_ds, test_ds = load_ds(cfg, os.path.join(DATA_DIR, "datasets"))    
    specs = train_ds.specs
    
    # load model
    datamodule = CLRSDataModule(train_dataset=train_ds,val_datasets=val_ds, test_datasets=test_ds, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS, test_batch_size=cfg.TEST.BATCH_SIZE)
    datamodule.val_dataloader()
    model = CLRSModel(specs=train_ds.specs, cfg=cfg)

    ckpt_dir = os.path.join(DATA_DIR, "checkpoints")
    train(model, datamodule, cfg, args.enable_wandb, train_ds.specs, seed = args.seed, checkpoint_dir=ckpt_dir)