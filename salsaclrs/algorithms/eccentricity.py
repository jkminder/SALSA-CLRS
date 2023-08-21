from clrs._src import probing
from clrs._src import specs
from ..specs import SPECS
import numpy as np


def eccentricity(A, source):
    # implementation of eccentricity from source node

    probes = probing.initialize(SPECS['eccentricity'])

    N = A.shape[0]
    A_pos = np.arange(N)

    
    # push input
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'adj': probing.graph(np.copy(A)),
            's': probing.mask_one(source, A.shape[0]),
            'A': np.copy(A),
        })
    
    flood_state = np.zeros(N, dtype=int)
    echo_state = np.zeros(N, dtype=int)
    msg_phase = np.zeros(N, dtype=bool)
    tree = np.zeros_like(A, dtype=bool)
    visited = np.zeros(N, dtype=bool)
    node_is_leaf = np.zeros(N, dtype=bool)

    def send_flood_msg(node, msg, state, tree):
        is_leaf = True
        for n in range(N):
            if A[node, n] and not visited[n]:
                state[n] = max(msg, state[n])
                tree[node, n] = True
                is_leaf = False
        return is_leaf

    def send_echo_msg(node, msg, state, tree, phase):
        for n in range(N):
            if tree[n, node] and visited[n]:
                # print(f'echo msg from {node} to {n}')
                state[n] = max(msg, state[n])
                tree[n, node] = False
                phase[n] = True

    probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'visited_h': np.copy(visited),
                'msgphase_h': np.copy(msg_phase),
                'tree_h': np.copy(tree),
                'floodstate_h': np.copy(flood_state),
                'echostate_h': np.copy(echo_state),
                'leaf_h': np.copy(node_is_leaf),
                'eccentricity_h': echo_state[source].copy(),
            })
    done = False
    while not done:
        next_visited = visited.copy()
        next_msg_phase = msg_phase.copy()
        next_tree = tree.copy()
        messages = np.zeros(N, dtype=int)
        next_echo_state = echo_state.copy()
        for node in range(N):
            if not msg_phase[node]:
                is_leaf = False
                # flood start
                if node == source and not visited[node]:
                    next_visited[node] = True
                    is_leaf = send_flood_msg(node, 1, messages, next_tree)

                # flood
                if flood_state[node] > 0 and not visited[node]:
                    # print("node {} is flooding '{}'".format(node, flood_state[node]+1))
                    next_visited[node] = True
                    is_leaf = send_flood_msg(node, flood_state[node] + 1, messages, next_tree)
                

                # print(f"F: Node {node} is leaf: {is_leaf}")
                if is_leaf:
                    # switch to echo phase and start echo
                    node_is_leaf[node] = True
                    next_echo_state[node] = flood_state[node]
                    # print(f"F: Node {node} is leaf and has echo state {flood_state[node]}")
                    next_msg_phase[node] = True
                    send_echo_msg(node, flood_state[node], next_echo_state, next_tree, next_msg_phase)
                    
            else:   
                if node_is_leaf[node]:
                    continue
                # echo -> check if message is received from all neighbors/is leaf
                is_leaf = True
                for neighbor in range(N):
                    if tree[node, neighbor] and not (tree[neighbor, node] and tree[node, neighbor]):
                        is_leaf = False

                if is_leaf:
                    # print(f"Node {node} is leaf and has echo state {echo_state[node]}")
                    if node == source:
                        done = True 
                        break
                    node_is_leaf[node] = True
                    # send echo back to parent
                    send_echo_msg(node, echo_state[node], next_echo_state, next_tree, next_msg_phase)

        visited = next_visited      
        msg_phase = next_msg_phase
        tree = next_tree

        # receive flood messages
        for node in range(N):
            if messages[node] > 0:
                if not visited[node]:
                    flood_state[node] = messages[node]
                else:
                    # check if message is received from all neighbors
                    is_leaf = True
                    for neighbor in range(N):
                        if tree[node, neighbor] and not (tree[neighbor, node] and tree[node, neighbor]):
                            is_leaf = False
                    if is_leaf:
                                        # switch to echo phase and start echo
                        node_is_leaf[node] = True
                        next_echo_state[node] = flood_state[node]
                        # print(f"F: Node {node} is leaf and has echo state {flood_state[node]}")
                        next_msg_phase[node] = True
                        send_echo_msg(node, flood_state[node], next_echo_state, next_tree, next_msg_phase)
        echo_state = next_echo_state


        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'visited_h': np.copy(visited),
                'msgphase_h': np.copy(msg_phase),
                'tree_h': np.copy(tree),
                'floodstate_h': np.copy(flood_state),
                'echostate_h': np.copy(echo_state),
                'leaf_h': np.copy(node_is_leaf),
                'eccentricity_h': echo_state[source].copy(),
            })
        
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'eccentricity': echo_state[source].copy(),
    })
    
    probing.finalize(probes)

    return echo_state[source].copy(), probes   


def eccentricity_path(A, source):
    # implementation of eccentricity from source node
    # path version - output is the path from source to the furthest node

    probes = probing.initialize(SPECS['eccentricity_path'])

    N = A.shape[0]
    A_pos = np.arange(N)

    
    # push input
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'adj': probing.graph(np.copy(A)),
            's': probing.mask_one(source, A.shape[0]),
            'A': np.copy(A),
        })
    
    flood_state = np.zeros(N, dtype=int)
    echo_state = np.zeros(N, dtype=int)
    msg_phase = np.zeros(N, dtype=bool)
    tree = np.zeros_like(A, dtype=bool)
    visited = np.zeros(N, dtype=bool)
    node_is_leaf = np.zeros(N, dtype=bool)
    eccentricity_path = np.zeros_like(A, dtype=bool)

    def send_flood_msg(node, msg, state, tree):
        is_leaf = True
        for n in range(N):
            if A[node, n] and not visited[n]:
                state[n] = max(msg, state[n])
                tree[node, n] = True
                is_leaf = False
        return is_leaf

    def send_echo_msg(node, msg, state, tree, phase):
        for n in range(N):
            if tree[n, node] and visited[n] and not tree[node, n]:
                if msg > state[n]:
                    state[n] = msg
                    eccentricity_path[n, :] = False
                    eccentricity_path[n, node] = True
                elif msg == state[n]:
                    #get index of previous best
                    index = np.where(eccentricity_path[n, :])[0]
                    if not len(index) or index[0] < node:
                        eccentricity_path[n, :] = False
                        eccentricity_path[n, node] = True
                    pass
                tree[n, node] = False
                phase[n] = True

    probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'visited_h': np.copy(visited),
                'msgphase_h': np.copy(msg_phase),
                'tree_h': np.copy(tree),
                'floodstate_h': np.copy(flood_state),
                'echostate_h': np.copy(echo_state),
                'leaf_h': np.copy(node_is_leaf),
                'eccentricitypath_h': np.copy(eccentricity_path),
            })
    done = False
    while not done:
        next_visited = visited.copy()
        next_msg_phase = msg_phase.copy()
        next_tree = tree.copy()
        messages = np.zeros(N, dtype=int)
        next_echo_state = echo_state.copy()
        for node in range(N):
            if not msg_phase[node]:
                is_leaf = False
                # flood start
                if node == source and not visited[node]:
                    next_visited[node] = True
                    is_leaf = send_flood_msg(node, 1, messages, next_tree)

                # flood
                if flood_state[node] > 0 and not visited[node]:
                    # print("node {} is flooding '{}'".format(node, flood_state[node]+1))
                    next_visited[node] = True
                    is_leaf = send_flood_msg(node, flood_state[node] + 1, messages, next_tree)
                

                # print(f"F: Node {node} is leaf: {is_leaf}")
                if is_leaf:
                    # switch to echo phase and start echo
                    node_is_leaf[node] = True
                    next_echo_state[node] = flood_state[node]
                    # print(f"F: Node {node} is leaf and has echo state {flood_state[node]}")
                    next_msg_phase[node] = True
                    send_echo_msg(node, flood_state[node], next_echo_state, next_tree, next_msg_phase)
                    
            else:   
                if node_is_leaf[node]:
                    continue
                # echo -> check if message is received from all neighbors/is leaf
                is_leaf = True
                for neighbor in range(N):
                    if tree[node, neighbor] and not (tree[neighbor, node] and tree[node, neighbor]):
                        is_leaf = False

                if is_leaf:
                    # print(f"Node {node} is leaf and has echo state {echo_state[node]}")
                    if node == source:
                        done = True 
                        break
                    node_is_leaf[node] = True
                    # send echo back to parent
                    send_echo_msg(node, echo_state[node], next_echo_state, next_tree, next_msg_phase)

        visited = next_visited      
        msg_phase = next_msg_phase
        tree = next_tree

        # receive flood messages
        for node in range(N):
            if messages[node] > 0:
                if not visited[node]:
                    flood_state[node] = messages[node]
                else:
                    # check if message is received from all neighbors
                    is_leaf = True
                    for neighbor in range(N):
                        if tree[node, neighbor] and not (tree[neighbor, node] and tree[node, neighbor]):
                            is_leaf = False
                    if is_leaf:
                                        # switch to echo phase and start echo
                        node_is_leaf[node] = True
                        next_echo_state[node] = flood_state[node]
                        # print(f"F: Node {node} is leaf and has echo state {flood_state[node]}")
                        next_msg_phase[node] = True
                        send_echo_msg(node, flood_state[node], next_echo_state, next_tree, next_msg_phase)
        echo_state = next_echo_state


        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'visited_h': np.copy(visited),
                'msgphase_h': np.copy(msg_phase),
                'tree_h': np.copy(tree),
                'floodstate_h': np.copy(flood_state),
                'echostate_h': np.copy(echo_state),
                'leaf_h': np.copy(node_is_leaf),
                'eccentricitypath_h': np.copy(eccentricity_path),
            })
        
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            'eccentricitypath': np.copy(eccentricity_path),
    })
    
    probing.finalize(probes)

    return np.copy(eccentricity_path), probes   