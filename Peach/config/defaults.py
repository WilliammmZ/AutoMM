from .config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config Definition
# -----------------------------------------------------------------------------
_C = CN()
# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1
# the root of the log file
_C.LOG_DIR = '/workspace/automm/Projects/VQE/log'
# the exp output will save in this file
_C.EXP_NAME = 'init_test'
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = True




# ---------------------------------------------------------------------------- #
#  Model Options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = "" # the path saves model weights

# Model Arch Found by name
_C.MODEL.META_ARCHITECTURE = "ExampleNet"




# ---------------------------------------------------------------------------- #
#  Backbone Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# Backbone Name
_C.MODEL.BACKBONE.NAME = 'build_example_backbone'




# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()


#--- BaseOption ---
# Number of samples per batch across all machines. This is also the number
# of training samples per step (i.e. per iteration). If we use 16 GPUs
# and SAMPLE_PER_BATCH = 32, each GPU will see 2 samples per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.SOLVER.SAMPLE_PER_BATCH = 16
_C.SOLVER.MAX_ITER = 40000
# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000


# ---- Optimizer ----
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False #SGD需要
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.FOR_EACH = True #加速for循环，但需要更多的显存


# ---- AMP ----
# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.AMP = CN({"ENABLED": False})


# ---- Gradient clipping ----
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0



# ---- lr scheduler ----
# Options: WarmupMultiStepLR / WarmupCosineLR / WarmupStepWithFixedGammaLR
# See Peach/solver/build.py for definition.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)
# The end lr, only used by WarmupCosineLR
_C.SOLVER.BASE_LR_END = 0.0
# For fixed_gama lr schedule
_C.SOLVER.NUM_DECAYS = 0


# ---- warmup ----
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 0 #not start
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.RESCALE_INTERVAL=False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

#MFQEV2_TRAIN/MFQEV2_VAL
#For MFQEV2 data we save the u/v channel for the test raw data 
_C.DATASETS.TRAIN = ('MFQEV2_TRAIN')  # MFQEV2_TRAINPP
_C.DATASETS.TEST = ('MFQEV2_TEST') #MFQEV2_TESTPP
_C.DATASETS.TEST_BATCH = 4 # This is world batch



# -----------------------------------------------------------------------------
#  DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4


# ---------------------------------------------------------------------------- #
#  Test Option
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# ---- batchnormal test ----
_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200

# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0
_C.TEST.PRE_EVAL = False
_C.TEST.SAVE_PATH = None

