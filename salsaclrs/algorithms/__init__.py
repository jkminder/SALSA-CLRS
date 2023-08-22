from clrs._src import algorithms
from .mis import fast_mis_2
from .eccentricity import eccentricity, eccentricity_path

LOCAL_ALGORITHMS = {
    'fast_mis': fast_mis_2,
    'eccentricity': eccentricity,
    'eccentricity_path': eccentricity_path
}

def get_algorithm(name):
    if name in LOCAL_ALGORITHMS:
        return LOCAL_ALGORITHMS[name]
    else:
        return getattr(algorithms, name)