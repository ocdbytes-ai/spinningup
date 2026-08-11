import os
import os.path as osp

# Default neural network backend for each algo.
# (Must be either 'tf1' or 'pytorch'. Note: the tf1 backend is legacy and only
# works if a compatible TensorFlow 1.x is installed separately; PyTorch is the
# supported default. TRPO only ever had a tf1 implementation, so requesting it
# will raise NotImplementedError until a PyTorch port lands.)
DEFAULT_BACKEND = {
    'vpg': 'pytorch',
    'trpo': 'pytorch',
    'ppo': 'pytorch',
    'ddpg': 'pytorch',
    'td3': 'pytorch',
    'sac': 'pytorch'
}

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = 5