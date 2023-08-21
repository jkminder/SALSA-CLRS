# 💃 SALSA-CLRS 💃

Implementation of "SALSA-CLRS: A Sparse and Scalable Benchmark for Algorithmic Reasoning". SALSA-CLRS is an extension around the [original clrs package](https://github.com/deepmind/clrs). It focusses on the algorithms that can be sparsified and work on sparse graphs. 
It provides pytorch based pyG datasets and dataloaders. It uses [loguru](https://loguru.readthedocs.io) for logging.

# Installation
```
cd SALSA-CLRS
pip install . 
```

# Examples

Available algorithms: `bfs`, `dfs`, `dijkstra`, `mst_prim`, `fast_mis_2`, `eccentricity`

A BFS train dataset with 10000 samples on "er" graphs with n in [16, 32] and p sampled from the range (0.1,0.3).
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="er", graph_generator_kwargs={"n": [16, 32], "p_range": (0.1, 0.3)}, hints=True)
```

A BFS train dataset with 10000 samples on "ws" graphs with n in [16, 32], k in [2,4,6] and p sampled in the range of (0.1,0.3).
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="ws", graph_generator_kwargs={"n": [16, 32], "k": [2,4,6], "p_range": (0.1, 0.3)}, hints=True)
```


A MST train dataset with 10000 samples on "delaunay" graphs with n in [16, 32].
```python
from salsaclrs import SALSACLRSDataset
ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="mst_prim", num_samples=10000, graph_generator="delaunay", graph_generator_kwargs={"n": [16, 32]}, hints=True)
```

Due to the hints you need to use the provided `CLRSDataLoader` instead of the default pyG `DataLoader`. This makes sure that batches are correctly collated. The API stays exactly the same.
```python
from salsaclrs import CLRSDataLoader
dl = CLRSDataLoader(ds, batch_size=32, workers=...)
```

Due to the hints you need to use the provided `CLRSDataLoader` instead of the default pyG `DataLoader`. This makes sure that batches are correctly collated. The API stays exactly the same.
```python
from salsaclrs import CLRSDataLoader
dl = CLRSDataLoader(ds, batch_size=32, workers=...)
```

# Pytorch Lightning

The library provides a pytorch lightning datamodule, that works with `SALSACLRSDataset` datasets as well as `DynamicDataset`s. It supports multiple validation and test datasets.

Example:
Train the model for the first 10 epochs with smaller graphs and after epoch 10 increase the graph size.
```python
from salsaclrs import SALSACLRSDataset, CLRSDataModule
import lightning.pytorch as pl

ds_train = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=10000, graph_generator="er", ignore_all_hints=False, hints=True, graph_generator_kwargs={"n": [16,32], "p": [0.1, 0.2,0.3]})

ds_val = SALSACLRSDataset(root=DATA_DIR, split="val", algorithm="bfs", num_samples=100, graph_generator="er", ignore_all_hints=False, hints=True,graph_generator_kwargs={"n": [32], "p": [0.1, 0.2,0.3]})
ds_test_small = SALSACLRSDataset(root=DATA_DIR, split="val", algorithm="bfs", num_samples=100, graph_generator="er", ignore_all_hints=False, hints=True, graph_generator_kwargs={"n": [32], "p": [0.1, 0.2,0.3]})
ds_test_large = SALSACLRSDataset(root=DATA_DIR, split="val", algorithm="bfs", num_samples=100, graph_generator="er", ignore_all_hints=False, hints=True, graph_generator_kwargs={"n": [128], "p": [0.1, 0.2,0.3]})

data_module = CLRSDataModule(train_dataset=ds_train, val_datasets=[ds_val], test_datasets=[ds_test_small, ds_test_large])
...
trainer = pl.Trainer(
        ...
    )
trainer.fit(model, data_module)
```