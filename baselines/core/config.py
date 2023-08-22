from yacs.config import CfgNode as CN


_C = CN()
_C.ALGORITHM = "bfs"
_C.RUN_NAME = "test"
# -----------------------------------------------------------------------------
# Model

_C.MODEL = CN()
_C.MODEL.HIDDEN_DIM = 128
_C.MODEL.MSG_PASSING_STEPS = 1

_C.MODEL.PROCESSOR = CN()
_C.MODEL.PROCESSOR.LAYERNORM = CN()
_C.MODEL.PROCESSOR.LAYERNORM.ENABLE = False
_C.MODEL.PROCESSOR.LAYERNORM.MODE = "graph"
_C.MODEL.PROCESSOR.NAME = "GINConv"

_C.MODEL.PROCESSOR.KWARGS = [{}] # dict not allowed so we use list of dict and just first element is used

_C.MODEL.GRU = CN()
_C.MODEL.GRU.ENABLE = False

_C.MODEL.DECODER_USE_LAST_HIDDEN = False
_C.MODEL.PROCESSOR_USE_LAST_HIDDEN = False

# -----------------------------------------------------------------------------
# Training

_C.TRAIN = CN()
_C.TRAIN.PRECISION = "16-mixed"
_C.TRAIN.ENABLE = True
_C.TRAIN.LOAD_CHECKPOINT = None

_C.TRAIN.BATCH_SIZE = 512
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.MAX_EPOCHS = 200
_C.TRAIN.GRADIENT_CLIP_VAL = 1.0
_C.TRAIN.EARLY_STOPPING_PATIENCE = 200

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
_C.TRAIN.OPTIMIZER.LR = 1e-3

_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.NAME = "ReduceLROnPlateau"
_C.TRAIN.SCHEDULER.ENABLE = False
_C.TRAIN.SCHEDULER.PARAMS = [{"mode": "min", "factor": 0.1, "patience": 10, "verbose": True}]

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT = 1.0
_C.TRAIN.LOSS.HINT_LOSS_WEIGHT = 1.0
_C.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT = 0.0
_C.TRAIN.LOSS.HIDDEN_LOSS_TYPE = "l2"


# -----------------------------------------------------------------------------
# Testing

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 8


# -----------------------------------------------------------------------------
# Data

_C.DATA = CN()

_C.DATA.ROOT = "./data"

_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.NUM_SAMPLES = 10000
_C.DATA.TRAIN.GRAPH_GENERATOR = ["er"]
_C.DATA.TRAIN.GENERATOR_PARAMS = [{"p": [0.5], "n": 16}]
_C.DATA.TRAIN.START_EPOCH = [0]

_C.DATA.VAL = CN()
_C.DATA.VAL.NUM_SAMPLES = 1000
_C.DATA.VAL.GRAPH_GENERATOR = "er"
_C.DATA.VAL.GENERATOR_PARAMS = [{"p": 0.5, "n": 16}]
_C.DATA.VAL.NICKNAME = "er_mid"

_C.DATA.TEST = CN()
_C.DATA.TEST.NUM_SAMPLES = [1000, 1000, 1000, 1000, 1000]
_C.DATA.TEST.GRAPH_GENERATOR = ["er", "er"]
_C.DATA.TEST.NICKNAME = ["er_mid", "er_hard"]
_C.DATA.TEST.GENERATOR_PARAMS = [{"p": [0.5, 0.6, 0.7, 0.8, 0.9], "n": 16}, {"p": [0.1, 0.2, 0.3, 0.4, 0.5], "n":16}]


# -----------------------------------------------------------------------------
# Logging

_C.LOGGING = CN()

_C.LOGGING.WANDB = CN()
_C.LOGGING.WANDB.PROJECT = "salsa-clrs"
_C.LOGGING.WANDB.ENTITY = ""
_C.LOGGING.WANDB.GROUP = ""


# -----------------------------------------------------------------------------

def get_cfg_defaults():
    return _C.clone()

def load_cfg(cfg_path):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    return cfg