# 💃 SALSA-CLRS 💃

Implementation of ["SALSA-CLRS: A Sparse and Scalable Benchmark for Algorithmic Reasoning"](https://arxiv.org/abs/2309.12253). SALSA-CLRS is an extension to the [original clrs package](https://github.com/deepmind/clrs), prioritizing scalability and the utilization of sparse representations. It provides pytorch based [PyG](https://www.pyg.org) datasets and dataloaders. It uses [loguru](https://loguru.readthedocs.io) for logging.

# Installation
If you want GPU support be sure to install [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://www.pyg.org) first. For the experiments you will also need the optional dependencies (torch_scatter, torch_sparse and torch_cluster). 
Then install SALSA-CLRS
```
pip install . 
```
Note: SALSA-CLRS depends on `dm-clrs` the CLRS Benchmark implementation, which depends on `jax`. This might take a while. If you get "No space left on device" Errors, install CLRS seperately with `pip install dm-clrs`.

# SALSA-CLRS Dataset

## Loading the dataset
With the following code snipped you can automatically download the datasets described in our paper.
The available algorithms: `bfs`, `dfs`, `dijkstra`, `mst_prim`, `fast_mis`, `eccentricity`

```python
from salsaclrs import load_dataset

train_dataset = load_dataset(algorithm="bfs", split="train", local_dir="path/to/local/data/store")
val_dataset = load_dataset(algorithm="bfs", split="val", local_dir="path/to/local/data/store")
# The test datasets are returned as a dictionary of datasets
test_datasets = load_dataset(algorithm="bfs", split="val", local_dir="path/to/local/data/store")
# E.g. get the ER test set on 16 nodes
er_16 = test_datasets["er_16"]
```
All of the returned objects are of type `SALSACLRSDataset`, a PyG dataset. With `ds.specs` you can get the types and specifications for an individual datapoint, similar to specs in CLRS. 

## Generating datasets

You can also generate new datasets according to your own requirements. A BFS train dataset with 10000 samples on "er" graphs with n in [16, 32] and p sampled from the range (0.1,0.3):
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="er", graph_generator_kwargs={"n": [16, 32], "p_range": (0.1, 0.3)}, hints=True)
```

A BFS train dataset with 10000 samples on "ws" graphs with n in [16, 32], k in [2,4,6] and p sampled in the range of (0.1,0.3):
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="ws", graph_generator_kwargs={"n": [16, 32], "k": [2,4,6], "p_range": (0.1, 0.3)}, hints=True)
```


A MST train dataset with 10000 samples on "delaunay" graphs with n in [16, 32]:
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="mst_prim", num_samples=10000, graph_generator="delaunay", graph_generator_kwargs={"n": [16, 32]}, hints=True)
```

When adding the flag `hints=False` the dataset will generate the hints, but not load them. If you want to generate a dataset without any hints, you can add the parameter `ignore_all_hints=True`. Please refer to the parameter descriptions of the classes for more detail.

## DataLoader

Due to the hints you need to use the provided `SALSACLRSDataLoader` instead of the default PyG `DataLoader`. This makes sure that batches are correctly collated. The API stays exactly the same.
```python
from salsaclrs import SALSACLRSDataLoader
dl = SALSACLRSDataLoader(ds, batch_size=32, num_workers=...)
```

## Pytorch Lightning

The library provides a pytorch lightning datamodule, that works with `SALSACLRSDataset` datasets. It supports multiple validation and test datasets.

Example:
```python
from salsaclrs import SALSACLRSDataset, SALSACLRSDataModule
import lightning.pytorch as pl

ds_train = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="er", ignore_all_hints=False, hints=True, graph_generator_kwargs={"n": [16,32], "p": [0.1, 0.2,0.3]})

ds_val = SALSACLRSDataset(root=DATA_DIR, split="val", algorithm="bfs", num_samples=100, graph_generator="er", ignore_all_hints=False, hints=True,graph_generator_kwargs={"n": [32], "p": [0.1, 0.2,0.3]})
ds_test_small = SALSACLRSDataset(root=DATA_DIR, split="val", algorithm="bfs", num_samples=100, graph_generator="er", ignore_all_hints=False, hints=True, graph_generator_kwargs={"n": [32], "p": [0.1, 0.2,0.3]})
ds_test_large = SALSACLRSDataset(root=DATA_DIR, split="val", algorithm="bfs", num_samples=100, graph_generator="er", ignore_all_hints=False, hints=True, graph_generator_kwargs={"n": [128], "p": [0.1, 0.2,0.3]})

data_module = SALSACLRSDataModule(train_dataset=ds_train, val_datasets=[ds_val], test_datasets=[ds_test_small, ds_test_large])
...
trainer = pl.Trainer(
        ...
    )
trainer.fit(model, data_module)
```


# Baselines

For the experiments a couple of more dependencies are required. Install them from `baselines/requirements.txt`. To rerun our experiments, run the `run_experiment.py` script in the `baselines` folder. You need to specify a seed and a data directory (the datasets and checkpoints will be stored there). You also need to specify an experiment configuration file, stored in the `configs` folder. The configuration file specifies the architecture, training details as well as the algorithm, e.g. for the GIN(E) experiment for dijkstra use `baselines/configs/dijkstra/GINE.yml`. Lastly, if you want to train with hints add the `--hints` flag to the script. If you want to log to [WANDB](https://wandb.ai) add the flag `--enable-wandb`, but be sure to specify your WANDB entity in the config file (`LOGGING.WANDB.ENTITY`). 
```bash
python baselines/run_experiment.py --cfg baselines/configs/dijkstra/GINE.yml --seed 42 --data-dir path/to/data/store --enable-wandb --hints
```
You can also run `python baselines/run_experiment.py --help` for more information. The results of the experiment will be logged to WANDB and saved to csv in a `results` folder in the project root.

## VRAM Usage

Evaluation of the VRAM usage of graphs up to 2 to the power of 15: 
For SALSA run:
```
python baselines/vram-usage/salsa-vram-usage.py --data_dir path/to/data/store
```
You can optionally specify a configuration file with `--cfg path/to/config.yml`.

For CLRS run: 
```
python baselines/vram-usage/clrs-vram-usage.py --processor_type triplet_mpnn
```

This will generate files `vram-usage-{clrs-{processor_type}, salsa}.csv` that specify the vram usage for each power of 2 in bytes.
