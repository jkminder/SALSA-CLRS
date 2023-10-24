""" This file contains custom data classes for CLRS data. It mainly concerns with with sparseing the graphs and converting them into pytorch geometric data objects.
Further it provides a custom dataloader that automatically pads hints to the maximum length of the batch. This is necessary because the hints are not padded in the dataset."""


import clrs
from typing import Any, Optional, List, Union
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import numpy as np
import os.path as osp
import os
from torch_geometric.data import Data, Dataset, Batch
import multiprocessing as mp
from collections import defaultdict
from huggingface_hub import hf_hub_download
import zipfile
import math

from tqdm.auto import tqdm, trange

import lightning.pytorch as pl

from loguru import logger

import torch

from .sampler import build_sampler, SAMPLERS

def er_probabilities(n):
    base = math.log(n) / n
    return (base * 1, base * 3)

__ws_k = [
    4,6,8
]


SALSA_CLRS_DATASETS = {
    "test": {
        "er_16": { "p_range": er_probabilities(16), "n": 16 },
        "er_80": { "p_range": er_probabilities(80), "n": 80 },
        "er_160": { "p_range": er_probabilities(160), "n": 160 },
        "er_800": { "p_range": er_probabilities(800), "n": 800 },
        "er_1600": { "p_range": er_probabilities(1600), "n": 1600 },
        "ws_16": { "p_range": (0.05, 0.2), "k": [4,6,8], "n": 16 },
        "ws_80": { "p_range": (0.05, 0.2), "k": [4,6,8], "n": 80 },
        "ws_160": { "p_range": (0.05, 0.2), "k": [4,6,8], "n": 160 },
        "ws_800": { "p_range": (0.05, 0.2), "k": [4,6,8], "n": 800 },
        "ws_1600": { "p_range": (0.05, 0.2), "k": [4,6,8], "n": 1600 },
        "delaunay_16": { "n": 16 },
        "delaunay_80": { "n": 80 },
        "delaunay_160": { "n": 160 },
        "delaunay_800": { "n": 800 },
        "delaunay_1600": { "n": 1600 },
    },
    "val": { "p_range": er_probabilities(16), "n": 16 },
    "train": {'p_range': er_probabilities(16), 'n':[4, 7, 11, 13, 16]}
}

def __dataset_available(algorithm, split, local_dir):
    if not osp.exists(osp.join(local_dir, algorithm, split)):
        return False
    
    datasets = os.listdir(osp.join(local_dir, algorithm, split))
    if split == "test" and len(datasets) < len(SALSA_CLRS_DATASETS["test"]):
        return False
    else:
        req_len = 10002 if split == "train" else 1002 # samples + 2 metadata files
        for dataset in datasets:
            if len(os.listdir(osp.join(local_dir, algorithm, split, dataset, "processed"))) != req_len:
                return False
        return True

def load_dataset(algorithm, split, local_dir):
    """Load the SALSA-CLRS dataset for the given algorithm and split.
    
    Args:
        algorithm (str): The algorithm to get the dataset for.
        split (str): The split to get the dataset for.
        local_dir (str): The directory to download the dataset to.
    """
    if algorithm not in SAMPLERS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available algorithms are {list(SAMPLERS.keys())}.")

    if split not in SALSA_CLRS_DATASETS:
        raise ValueError(f"Unknown split '{split}'. Available splits are {list(SALSA_CLRS_DATASETS.keys())}.")
    
    # check if the dataset is already downloaded
    if not __dataset_available(algorithm, split, local_dir):
        logger.info(f"Downloading dataset for algorithm '{algorithm}'...")
        hf_hub_download(repo_id="SALSA-CLRS/SALSA-CLRS", filename=f"{algorithm}.zip", repo_type="dataset", local_dir = local_dir, local_dir_use_symlinks=False)

        logger.info(f"Extracting dataset...")
        with zipfile.ZipFile(osp.join(local_dir, f"{algorithm}.zip"), 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        
    if split == "test":
        return {k: SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="test", algorithm=algorithm, num_samples=1000, graph_generator=k.split("_")[0], graph_generator_kwargs=SALSA_CLRS_DATASETS["test"][k], nickname=k) for k in SALSA_CLRS_DATASETS["test"]}
    elif split == "val":
        return SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="val", algorithm=algorithm, num_samples=1000, graph_generator="er", graph_generator_kwargs=SALSA_CLRS_DATASETS["val"])
    else:
        return SALSACLRSDataset(ignore_all_hints=False, root=local_dir, split="train", algorithm=algorithm, num_samples=10000, graph_generator="er", graph_generator_kwargs=SALSA_CLRS_DATASETS["train"])


class NotSparseError(Exception):
    """Raised when the data is not sparse."""
    pass

class CLRSData(Data):
    """A data object for CLRS data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def infer_type(dp_type, data):
    return data.astype(np.float32) # convert to float32

def verify_sparseness(data, edge_index, data_name):
    """Verify that the n by n data is sparse (meaning that it only contains values for edges)."""
    edge_mask = np.zeros_like(data, dtype=bool)
    edge_mask[edge_index[0], edge_index[1]] = True

    if np.any(data[~edge_mask] > 0):
        logger.error(data[edge_mask])
        raise NotSparseError(f"The data '{data_name}' is not sparse. It contains values for non-edges.")
    
def pointer_to_one_hot(pointer, n):
    """Convert a pointer to a one-hot vector."""
    return (np.arange(n) == pointer.reshape(-1, 1)).astype(float)
    # return (torch.arange(n) == pointer.unsqueeze(1)).float()

def to_torch(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor(value)
    
def to_sparse_data(inputs, hints, outputs, use_hints=True):
    data_dict = {}
    input_attributes = []
    hint_attributes = []
    output_attributes = []
    data_dict['length'] = hints[0].data.shape[0]
    # first get the edge index
    for dp in inputs:
        if dp.name == "adj":
            edge_index, _ = from_scipy_sparse_matrix(coo_matrix(dp.data[0]))
            data_dict['edge_index'] = edge_index
    # Parse inputs
    for dp in inputs:
        if dp.name == "adj":
            continue
        elif dp.name == "A":
            # add self loops
            unique_values = np.unique(dp.data[0])
            is_weighted = unique_values.size != 2 or not np.all(unique_values == np.array([0,1]))
            if is_weighted:
                data_dict["weights"] = infer_type("A", (dp.data[0] + np.eye(dp.data[0].shape[0]))[data_dict["edge_index"][0], data_dict["edge_index"][1]])
        elif dp.location == clrs.Location.EDGE:
            verify_sparseness(dp.data[0], data_dict["edge_index"], dp.name)
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
            input_attributes.append(dp.name)
        elif dp.location == clrs.Location.NODE:
            if dp.type_ == clrs.Type.POINTER:
                # Convert pointers to one-hot edge masks
                n = dp.data[0].shape[0]
                pointer_matrix = pointer_to_one_hot(dp.data[0], n)
                verify_sparseness(pointer_matrix, data_dict["edge_index"], dp.name)
                data_dict[dp.name] = pointer_matrix[data_dict["edge_index"][0], data_dict["edge_index"][1]]
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
            input_attributes.append(dp.name)
        else: # Graph
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
    # Parse outputs
    for dp in outputs:
        output_attributes.append(dp.name)
        if dp.location == clrs.Location.EDGE:
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0][data_dict["edge_index"][0], data_dict["edge_index"][1]])
        elif dp.location == clrs.Location.NODE:
            if dp.type_ == clrs.Type.POINTER:
                # Convert pointers to one-hot edge masks
                n = dp.data[0].shape[0]
                pointer_matrix = pointer_to_one_hot(dp.data[0], n)
                verify_sparseness(pointer_matrix, data_dict["edge_index"], dp.name)
                data_dict[dp.name] = pointer_matrix[data_dict["edge_index"][0], data_dict["edge_index"][1]]
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
        else: # Graph
            data_dict[dp.name] = infer_type(dp.type_, dp.data[0])
    if use_hints:
        # Parse hints
        for dp in hints:
            hint_attributes.append(dp.name)
            if dp.location == clrs.Location.EDGE or (dp.location == clrs.Location.NODE and dp.type_ == clrs.Type.POINTER):
                arr = dp.data.squeeze(1) # Hints, N, N, D (...)
                if dp.location == clrs.Location.NODE:
                    # arr is Hints, N, D (...)
                    # Convert pointers to one-hot edge masks
                    stages = []
                    for hd in range(arr.shape[0]):
                        n = arr.shape[1]
                        pointer_matrix = pointer_to_one_hot(arr[hd], n)
                        verify_sparseness(pointer_matrix, data_dict["edge_index"], dp.name)
                        stages.append(pointer_matrix)
                    
                    arr = np.stack(stages, axis=0)
                else:
                    # just verify sparseness
                    for hd in range(arr.shape[0]):
                        verify_sparseness(arr[hd], data_dict["edge_index"], dp.name)
                # Determine the number of dimensions of the array
                num_dims = arr.ndim
                transpose_indices = tuple(range(num_dims))
                transpose_indices = (1, 2, 0) + transpose_indices[3:]
                data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices)[data_dict["edge_index"][0], data_dict["edge_index"][1]])
            elif dp.location == clrs.Location.NODE and not dp.type_ == clrs.Type.POINTER:
                arr = dp.data.squeeze(1) # Hints, N, D (...)
                # Determine the number of dimensions of the array
                num_dims = arr.ndim
                # Create a tuple of indices to swap the first two dimensions
                transpose_indices = tuple(range(num_dims))
                transpose_indices = (1, 0) + transpose_indices[2:]
                data_dict[dp.name] = infer_type(dp.type_, arr.transpose(*transpose_indices))
            else:
                data_dict[dp.name] = infer_type(dp.type_, dp.data.squeeze(1)[np.newaxis, ...])

        
    data_dict = {k: to_torch(v) for k,v in data_dict.items()}
    data = CLRSData(**data_dict)    
    data.hints = hint_attributes
    data.inputs = input_attributes
    data.outputs = output_attributes
    return data

def _collapse_val(val):
    if (isinstance(val, list) or isinstance(val, tuple)) and isinstance(val[0], int) and val == list(range(val[0], val[-1]+1)):
        return f"range({val[0]}, {val[-1]+1})"
    elif (isinstance(val, list) or isinstance(val, tuple)) and isinstance(val[0], float):
        return str([f"{x:.2e}" for x in val])
    else:
        return str(val)

class SALSACLRSDataset(Dataset):
    def __init__(self, root, 
                 split,
                 algorithm,
                 num_samples = 1000,
                 verify_duplicates = False,
                 forbidden_duplicates_datasets = [],
                 hints=True, ignore_all_hints=False, nickname=None, graph_generator="er", graph_generator_kwargs={"n": 16, "p": 0.1},max_cores=-1, **kwargs):
        """ Dataset for SALSA CLRS problems.

        Args:
            root (str): Root directory where the dataset should be saved.
            split (str): Split of the dataset to use. One of ['train', 'val', 'test'].
            algorithm (str): Algorithm to use. Check salsa-clrs.ALGORITHMS for a list of available algorithms.
            num_samples (int): Number of samples to collect.
            hints (bool): Whether to use hints or not (hints are still loaded but not returned in the data dict)
            ignore_all_hints (bool): Whether to ignore all hints or not. If True, hints are not even generated, might be beneficial for memory.
            verify_duplicates (bool): Whether to verify that no duplicates are generated (on graph level - no two graphs are the same)
            forbidden_duplicates_datasets (list): List of datasets for which duplicates are forbidden. This is useful when you want to combine multiple datasets but do not want to have duplicates between them.
            nickname (str): Optional nickname for the dataset (mainly intended for logging purposes).
            graph_generator (str): Name of the graph generator to use. 
            graph_generator_kwargs (dict): Keyword arguments to pass to the graph generator.
            max_cores (int): Maximum number of cores to use for multiprocessing. If -1, it is serial. If None, it is the number of cores on the machine (default: -1)
            **kwargs: Keyword arguments to pass to the algorithm sampler.
        """
        self.algorithm = algorithm
        self.num_samples = num_samples
        self.graph_generator = graph_generator
        self.graph_generator_kwargs = graph_generator_kwargs
        self.max_cores = max_cores if max_cores is not None else mp.cpu_count()
        
        self.sampler, self.specs = build_sampler(algorithm, graph_generator, graph_generator_kwargs, **kwargs)

        self.verify_duplicates = verify_duplicates
        self.forbidden_duplicates_datasets = forbidden_duplicates_datasets
        name = "graphgenerator=" + graph_generator + '_' + '_'.join([f'{key}={_collapse_val(val)}' for key, val in graph_generator_kwargs.items()])
        self.hints = hints
        self.ignore_all_hints = ignore_all_hints
        if ignore_all_hints:
            name += "-nohints"

        self.nickname = nickname
        root = osp.join(root, algorithm, split, name)
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
        super().__init__(root, None, None, None)
        
        self._update_specs()

    def _update_specs(self):
        # get a batch
        batch = self.get(0)
        specs = {}
        for key, data in self.specs.items():
            if key not in batch:
                continue
            stage, location, type_ = data
            if type_ == clrs.Type.CATEGORICAL:
                specs[key] = (stage, location, clrs.Type.CATEGORICAL, batch[key].shape[-1])
            else:
                specs[key] = (stage, location, type_, None)
        self.specs = specs

    def _is_duplicate(self, data, dataset):
        for idx in range(len(dataset)):
            if (data.edge_index == dataset[idx]).all().item():
                return True
        return False

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.num_samples)]

    def _get_sample(self, idx):
        inp, outp, hints = self.sampler.next()
        data = to_sparse_data(inp, hints, outp, not self.ignore_all_hints)
        return data

    def process(self):
        logger.info(f"Generating {self.algorithm} dataset with {self.num_samples} samples on {self.max_cores} cores")
        logger.info(f"Graph generator: {self.graph_generator} with kwargs {self.graph_generator_kwargs}")

        pb = tqdm(total=self.num_samples)
        generated_data = []
        i = 0

        if self.verify_duplicates:
            _dataset = defaultdict(list) # dataset for verification of duplicates
            # pre-load forbidden datasets
            if len(self.forbidden_duplicates_datasets) > 0:
                logger.info("Loading forbidden datasets for duplicate verification")
            for dataset in self.forbidden_duplicates_datasets:
                for idx in range(len(dataset)):
                    _dataset[dataset[idx].edge_index.shape].append(dataset[idx].edge_index)

        if self.max_cores != -1:
            # We don't compute the full batch at once to safe on memory (mainly relevant for hints and long algo iterations)
            with mp.Pool(processes=self.max_cores) as pool: 
                while i < self.num_samples:
                    for data in pool.imap_unordered(self._get_sample, range(min(500, self.num_samples - len(generated_data))), chunksize=1):
                        generated_data.append(data)
                        pb.update(1)
    
                    for idx in range(len(generated_data)):
                        while (True):
                            data = generated_data[idx]
                            if self.verify_duplicates:
                                if self._is_duplicate(data, _dataset[data.edge_index.shape]):
                                    inp, outp, hints = self.sampler.next()
                                    generated_data[idx] = to_sparse_data(inp, hints, outp, not self.ignore_all_hints)
                                    continue
                                else:
                                    _dataset[data.edge_index.shape].append(data.edge_index)
                                    break
                            else:
                                break

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        torch.save(data, osp.join(self.processed_dir, f'data_{i}.pt'))
                        i += 1

                    generated_data = []
        else:
            while i < self.num_samples:
                for data in map(self._get_sample, range(min(500, self.num_samples - len(generated_data)))):
                    generated_data.append(data)
                    pb.update(1)
   
                for idx in range(len(generated_data)):
                    while (True):
                        data = generated_data[idx]
                        if self.verify_duplicates:
                            if self._is_duplicate(data, _dataset[data.edge_index.shape]):
                                inp, outp, hints = self.sampler.next()
                                generated_data[idx] = to_sparse_data(inp, hints, outp, not self.ignore_all_hints)
                                continue
                            else:
                                _dataset[data.edge_index.shape].append(data.edge_index)
                                break
                        else:
                            break

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, osp.join(self.processed_dir, f'data_{i}.pt'))
                    i += 1

                generated_data = []
                
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        if not self.hints and not self.ignore_all_hints:
            for hint in data.hints:
                delattr(data, hint)
            del data.hints
        return data

class DynamicDataset:
    def __init__(self) -> None:
        """
        A dataset that can change over time. This is usefull when the dataset changes over time, e.g. when the number of nodes changes.
        """
        self.datasets = []
        self.current_epoch = 0
        self.current_dataset = None
        self.next_change = None
        self.current_idx = 0
        self.specs = None
        self.reload_every_n_epochs = 0

    def add_dataset(self, start_epoch, dataset):
        self.datasets.append((start_epoch, dataset))
        self.datasets = sorted(self.datasets, key=lambda x: x[0])
        self.specs = dataset.specs
        self.current_dataset = self.datasets[0][1]

        if len(self.datasets) > 1:
            self.reload_every_n_epochs = np.gcd.reduce([x[0] for x in self.datasets[1:]]).item()
            self.next_change = self.datasets[1][0]

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _update_dataset(self):
        if self.next_change is not None and self.current_epoch >= self.next_change:
            logger.info(f"Changing dataset at epoch {self.current_epoch}")
            self.current_idx += 1
            self.current_dataset = self.datasets[self.current_idx][1]
            if self.current_idx < len(self.datasets) - 1:
                self.next_change = self.datasets[self.current_idx+1][0]
            else:
                self.next_change = None

    def get(self):
        self._update_dataset()
        return self.current_dataset

class CLRSCollater(object):
    """Special Collater that can handle hints. """
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def normalise_length(self, batch):
        """Normalise the length of the batch by padding with zeros."""
        max_len = max([data.length for data in batch])
        for data in batch:
            if data.length < max_len:
                # pad all hints
                for hint in data.hints:
                    data[hint] = torch.cat([data[hint], torch.zeros(*data[hint].shape[:1], max_len - data[hint].shape[1], *data[hint].shape[2:])], dim=1)
                # pad randomness
                if "randomness" in data.inputs:
                    data["randomness"] = torch.cat([data["randomness"], torch.zeros(*data["randomness"].shape[:1], max_len - data["randomness"].shape[1], *data["randomness"].shape[2:])], dim=1)

        return batch
    

    def collate(self, batch):
        if "hints" in batch[0].keys() or "randomness" in batch[0].inputs:
            batch = self.normalise_length(batch)
        batch = Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        batch.hints = batch.hints[0]
        batch.inputs = batch.inputs[0]
        batch.outputs = batch.outputs[0]
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class SALSACLRSDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch..
pip install importlib-resources
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, List[Data]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=CLRSCollater(follow_batch,
                                             exclude_keys), **kwargs)
        

class SALSACLRSDataModule(pl.LightningDataModule):
    """A Lightning DataModule for the CLRS dataset."""
    def __init__(self, train_dataset=None, val_datasets=None, test_datasets=None, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        if isinstance(train_dataset, DynamicDataset):
            self.reload_every_n_epochs = train_dataset.reload_every_n_epochs
        else:
            self.reload_every_n_epochs = 0

        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        if not isinstance(test_datasets, list):
            self.test_datasets = [test_datasets]
        self.val_datasets = val_datasets

        self.test_batch_size = kwargs.pop("test_batch_size", None)
        self.kwargs = kwargs

        self._val_dataloaders = None
    
    def get_val_loader_nickname(self, idx):
        if isinstance(self.val_datasets, list):
            name = self.val_datasets[idx].nickname
        else:
            name = self.val_datasets.nickname
        return name if name else idx
    
    def get_test_loader_nickname(self, idx):
        name = self.test_datasets[idx].nickname
        return name if name else idx
    
    def dataloader(self, dataset: Dataset, **kwargs) -> SALSACLRSDataLoader:
        return SALSACLRSDataLoader(dataset, **kwargs)
    
    def train_dataloader(self) -> SALSACLRSDataLoader:
        if self.reload_every_n_epochs > 0 or isinstance(self.train_dataset, DynamicDataset):
            self.train_dataset.set_epoch(self.trainer.current_epoch)
            ds = self.train_dataset.get()
        else:
            ds = self.train_dataset
        return self.dataloader(ds, shuffle=True, persistent_workers=True, **self.kwargs)
    
    def val_dataloader(self) -> SALSACLRSDataLoader:
        if self._val_dataloaders is None:
            if isinstance(self.val_datasets, list):
                self._val_dataloaders = [self.dataloader(val_dataset, shuffle=False, persistent_workers=True, **self.kwargs) for val_dataset in self.val_datasets]
            else:
                self._val_dataloaders = self.dataloader(self.val_datasets, shuffle=False, persistent_workers=True, **self.kwargs)
        return self._val_dataloaders
    
    def test_dataloader(self) -> SALSACLRSDataLoader:
        bs = self.test_batch_size
        kwargs = self.kwargs.copy()
        if bs is not None:
            kwargs["batch_size"] = bs
        
        kwargs["num_workers"] = 0 # we don't want to use multiprocessing for testing as there have been problems with shared memory
        return [self.dataloader(test_dataset, shuffle=False, **kwargs) for test_dataset in self.test_datasets]
    


