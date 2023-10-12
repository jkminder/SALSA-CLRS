from clrs._src import algorithms
from .mis import fast_mis_2
from .eccentricity import eccentricity

LOCAL_ALGORITHMS = {
    'fast_mis': fast_mis_2,
    'eccentricity': eccentricity,
}

def get_algorithm(name):
    if name in LOCAL_ALGORITHMS:
        return LOCAL_ALGORITHMS[name]
    else:
        return getattr(algorithms, name)