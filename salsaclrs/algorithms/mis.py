from clrs._src import probing
from clrs._src import specs
from ..specs import SPECS
import numpy as np

def fast_mis_2(A):
    # implementation of fast_mis_2
    N = A.shape[0]
    probes = probing.initialize(SPECS['fast_mis_2'])
    A_pos = np.arange(A.shape[0])


    A_in = np.copy(A)
    round = 0
    all_random_numbers = []
    in_mis = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)

    while alive.sum() > 0:
        # step 1: each node chooses a random number and sends it to all neighbors
        random_numbers = np.random.rand(N)
        all_random_numbers.append(random_numbers) # random numbers are fixed for the whole algorithm and every step
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'Ain_h': np.copy(A_in),
                'alive_h': np.copy(alive),
                'inmis_h': np.copy(in_mis),
                'phase_h': np.array(0)
            })

        round += 1
        # step 2 and 3 are merged in this implementation
        # step 2: each node compares its random number to those of its neighbors
        # and if it is the largest, it declares itself in the MIS and notifies its neighbors
        for node in range(N):
            if not alive[node]:
                continue
            my_r = random_numbers[node]
            mis = True
            for neighbor in range(N):
                if A_in[node, neighbor] == 1:
                    neighbor_r = random_numbers[neighbor]
                    if my_r > neighbor_r or (my_r == neighbor_r and node < neighbor):
                        mis = False
                        break
            if mis:
                in_mis[node] = True
        # step 3: if a node is in the MIS or has a neighbor in the MIS, it dies
        for node in range(N):
            if not alive[node]:
                continue
            if in_mis[node]:
                alive[node] = False
            else:
                for neighbor in range(N):
                    if A_in[node, neighbor] == 1 and in_mis[neighbor]:
                        alive[node] = False
                        break
        for node in range(N): # remove edges to dead nodes
            if not alive[node]:
                A_in[node, :] = 0
                A_in[:, node] = 0
        all_random_numbers.append(random_numbers) # random numbers are fixed for the whole algorithm and every step
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'Ain_h': np.copy(A_in),
                'alive_h': np.copy(alive),
                'inmis_h': np.copy(in_mis),
                'phase_h': np.array(1)
            })
        round += 1

    # push output
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'inmis': np.copy(in_mis),
        })

    # push input
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'adj': probing.graph(np.copy(A)),
            'randomness': np.array(all_random_numbers).T
        })
    
    probing.finalize(probes)

    return in_mis, probes