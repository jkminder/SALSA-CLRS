# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sampling utilities adapted from https://github.com/deepmind/clrs/blob/master/clrs/_src/samplers.py"""

import abc
import collections
import inspect
import types

from typing import Any, Callable, List, Optional, Tuple, Dict
from loguru import logger


from clrs._src import probing
from clrs._src import specs

import numpy as np
import networkx as nx
from tqdm import trange
from scipy.spatial import Delaunay

from .specs import SPECS
from .algorithms import get_algorithm

_Array = np.ndarray
_DataPoint = probing.DataPoint
Trajectory = List[_DataPoint]
Trajectories = List[Trajectory]


Algorithm = Callable[..., Any]
Features = collections.namedtuple('Features', ['inputs', 'hints', 'lengths'])
FeaturesChunked = collections.namedtuple(
    'Features', ['inputs', 'hints', 'is_first', 'is_last'])
Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])


class Sampler(abc.ABC):
  """Sampler abstract base class."""

  def __init__(
      self,
      algorithm: Algorithm,
      spec: specs.Spec,
      seed: Optional[int] = None,
      graph_generator: Optional[str] = None,
      graph_generator_kwargs: Optional[Dict[str, Any]] = None,
      **kwargs,
  ):
    """Initializes a `Sampler`.

    Args:
      algorithm: The algorithm to sample from
      spec: The algorithm spec.
      seed: RNG seed.
      **kwargs: Algorithm kwargs.
    """

    # Use `RandomState` to ensure deterministic sampling across Numpy versions.
    self._rng = np.random.RandomState(seed)
    self._graph_generator = graph_generator
    self._graph_generator_kwargs = graph_generator_kwargs
    self._spec = spec
    self._algorithm = algorithm
    self._kwargs = kwargs


  def _get_graph_generator_kwargs(self):
    return self._graph_generator_kwargs

  def next(self) -> Feedback:
    data = self._sample_data(**self._kwargs)
    _ , probes = self._algorithm(*data)
    inp, outp, hint = probing.split_stages(probes, self._spec)
    return inp, outp, hint


  def _create_graph(self, n, weighted, directed, low=0.0, high=1.0, **kwargs):
    """Create graph."""
    if self._graph_generator is None or self._graph_generator == 'er':
      mat =  self._random_er_graph(n=n, **kwargs)
    elif self._graph_generator == 'ws':
      mat =  self._watt_strogatz_graph(n=n, **kwargs)
    elif self._graph_generator == 'grid':
      mat =  self._grid_graph(n=n, **kwargs)
    elif self._graph_generator == 'delaunay':
      mat =  self._random_delaunay_graph(n=n, **kwargs)
    elif self._graph_generator == 'path':
      mat =  self._path_graph(n=n, **kwargs)
    elif self._graph_generator == 'tree':
      mat =  self._tree_graph(n=n, **kwargs)
    elif self._graph_generator == 'complete':
      mat = np.ones((n,n))
    else:
      raise ValueError(f'Unknown graph generator {self._graph_generator}.')
    n = mat.shape[0]
    assert not directed, 'Directed graphs not supported yet.'
    if weighted:
      weights = self._rng.uniform(low=low, high=high, size=(n, n))
      if not directed:
        weights *= np.transpose(weights)
        weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
      mat = mat.astype(float) * weights
    return mat
  
  @abc.abstractmethod
  def _sample_data(self, *args, **kwargs) -> List[_Array]:
    pass

  def _select_parameter(self, parameter, parameter_range=None):
    if parameter_range is not None:
      assert len(parameter_range) == 2
      return self._rng.uniform(*parameter_range)
    if isinstance(parameter, list) or isinstance(parameter, tuple):
      return self._rng.choice(parameter)
    else:
      return parameter

  def _random_er_graph(self, n, p=None, p_range=None, directed=False, acyclic=False, connected=True,
                       weighted=False, low=0.0, high=1.0, *args, **kwargs):
    """Random Erdos-Renyi graph."""
    n = self._select_parameter(n)
    p = self._select_parameter(p, p_range)

    while True:
      g = nx.erdos_renyi_graph(n, p, directed=directed)
      if connected:
        # ensure that the graph is connected
        if not nx.is_connected(g):
          continue
      return nx.to_numpy_array(g)

  def _watt_strogatz_graph(self, n, k, *args, p=None, p_range=None, **kwargs):
    """Watts-Strogatz graph."""
    n = self._select_parameter(n)
    k = self._select_parameter(k)
    p = self._select_parameter(p, p_range)
    g = nx.connected_watts_strogatz_graph(n, k, p)
    mat =  nx.to_numpy_array(g)
    return mat

  def _path_graph(self, n, *args, **kwargs):
    """Path graph."""
    g = nx.path_graph(n)
    return nx.to_numpy_array(g)

  def _grid_graph(self, dimensions, n, *args, **kwargs):
    """2D grid graph."""
    if n != np.prod(dimensions):
      raise ValueError(f'Grid dimensions {dimensions} do not match n={n}.')
    mat = nx.to_numpy_array(nx.grid_graph(dimensions))
    return mat
  
  def _random_delaunay_graph(self, n, *args, **kwargs):
    """Random delaunay graph."""
    n = self._select_parameter(n)
    # sample n points in space
    points = np.random.rand(n, 2)
    # create a delaunay triangulation
    tri = Delaunay(points)
    # create a networkx graph
    G = nx.Graph()
    # add the points as nodes
    G.add_nodes_from(range(n))
    # add the edges
    for edge in tri.simplices:
        G.add_edge(edge[0], edge[1])
        G.add_edge(edge[1], edge[2])
        G.add_edge(edge[2], edge[0])
    return nx.to_numpy_array(G)

  def _tree_graph(self, n, r, *args, **kwargs):
    """Tree."""
    n = self._select_parameter(n)
    r = self._select_parameter(r)
    if n < 2:
      raise ValueError(f'Cannot generate tree of size {n}.')
    mat = np.zeros((n, n))
    for i in range(1, n):
        mat[i, (i - 1) // r] = 1
    # make symmetric
    mat = mat + mat.T
    mat = mat.astype(bool).astype(int)
    return mat



class DfsSampler(Sampler):
  """DFS sampler."""

  def _sample_data(self):
    generator_kwargs = self._get_graph_generator_kwargs()
    generator_kwargs.update({"directed": False, "acyclic": False, "weighted": False})
    graph = self._create_graph(**generator_kwargs)
    return [graph]


class BfsSampler(Sampler):
  """BFS sampler."""

  def _sample_data(self):
    generator_kwargs = self._get_graph_generator_kwargs()
    generator_kwargs.update({"directed": False, "acyclic": False, "weighted": False})
    graph = self._create_graph(**generator_kwargs)
    source_node = self._rng.choice(graph.shape[0])
    return [graph, source_node]

class ArticulationSampler(Sampler):
  """Articulation Point sampler."""

  def _sample_data(self):
    generator_kwargs = self._get_graph_generator_kwargs()
    generator_kwargs.update({"directed": False, "acyclic": False, "weighted": False})
    graph = self._create_graph(**generator_kwargs)
    return [graph]


class MSTSampler(Sampler):
  """MST sampler for Kruskal's algorithm."""

  def _sample_data(self, low=0.0, high=1.0):
    generator_kwargs = self._get_graph_generator_kwargs()
    generator_kwargs.update({"directed": False, "acyclic": False, "weighted": True, "low": low, "high": high})  
    graph = self._create_graph(**generator_kwargs)
    return [graph]


class BellmanFordSampler(Sampler):
  """Bellman-Ford sampler."""

  def _sample_data(self, low=0.0, high=1.0):
    generator_kwargs = self._get_graph_generator_kwargs()
    generator_kwargs.update({"directed": False, "acyclic": False, "weighted": True, "low": low, "high": high})  
    graph = self._create_graph(**generator_kwargs)
    source_node = self._rng.choice(graph.shape[0])
    return [graph, source_node]

class MISSampler(Sampler):
  """MIS sampler for fast mis algos."""

  def _sample_data(self):
    generator_kwargs = self._get_graph_generator_kwargs()
    generator_kwargs.update({"directed": False, "acyclic": False, "weighted": False})  
    graph = self._create_graph(**generator_kwargs)
    return [graph]


def build_sampler(
    name: str,
    graph_generator: str = 'er',
    graph_generator_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Sampler, specs.Spec]:
  """Builds a sampler. See `Sampler` documentation."""
  if name not in SPECS or name not in SAMPLERS:
    raise NotImplementedError(f'No implementation of algorithm {name}.')
  spec = SPECS[name]
  algorithm = get_algorithm(name)
  sampler_class = SAMPLERS[name]
  # Ignore kwargs not accepted by the sampler.
  sampler_args = inspect.signature(sampler_class._sample_data).parameters  # pylint:disable=protected-access
  clean_kwargs = {k: kwargs[k] for k in kwargs if k in sampler_args}

  if set(clean_kwargs) != set(kwargs):
    logger.warning(f'Ignoring kwargs {set(kwargs).difference(clean_kwargs)} when building sampler class {sampler_class}')
  sampler = sampler_class(algorithm, spec, seed=seed, graph_generator=graph_generator, graph_generator_kwargs=graph_generator_kwargs,
                          **clean_kwargs)
  return sampler, spec



SAMPLERS = {
    'dfs': DfsSampler,
    # 'articulation_points': ArticulationSampler,
    # 'bridges': ArticulationSampler,
    'bfs': BfsSampler,
    # 'mst_kruskal': MSTSampler,
    'mst_prim': BellmanFordSampler,
    # 'bellman_ford': BellmanFordSampler,
    'dijkstra': BellmanFordSampler,
    'fast_mis': MISSampler,
    'eccentricity': BfsSampler,
    # 'eccentricity_path': BfsSampler,
}

