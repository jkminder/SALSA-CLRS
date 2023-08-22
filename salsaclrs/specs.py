from clrs._src.specs import Stage, Location, Type, SPECS

SPECS = dict(SPECS)
SPECS.update({
    'fast_mis': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'randomness': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'Ain_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'alive_h': (Stage.HINT, Location.NODE, Type.MASK),
        'phase_h': (Stage.HINT, Location.GRAPH, Type.MASK),
        'inmis_h': (Stage.HINT, Location.NODE, Type.MASK),
        'inmis': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
    },
    'eccentricity': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'visited_h': (Stage.HINT, Location.NODE, Type.MASK),
        'msgphase_h': (Stage.HINT, Location.NODE, Type.MASK),
        'floodstate_h': (Stage.HINT, Location.NODE, Type.SCALAR),
        'echostate_h': (Stage.HINT, Location.NODE, Type.SCALAR),
        'tree_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'leaf_h': (Stage.HINT, Location.NODE, Type.MASK),
        'eccentricity_h': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'eccentricity': (Stage.OUTPUT, Location.GRAPH, Type.SCALAR),
    },
    'eccentricity_path': {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'visited_h': (Stage.HINT, Location.NODE, Type.MASK),
        'msgphase_h': (Stage.HINT, Location.NODE, Type.MASK),
        'floodstate_h': (Stage.HINT, Location.NODE, Type.SCALAR),
        'echostate_h': (Stage.HINT, Location.NODE, Type.SCALAR),
        'tree_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'leaf_h': (Stage.HINT, Location.NODE, Type.MASK),
        'eccentricitypath_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'eccentricitypath': (Stage.OUTPUT, Location.EDGE, Type.MASK),
    }
})
