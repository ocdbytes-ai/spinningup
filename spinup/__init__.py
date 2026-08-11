# Algorithms (PyTorch backend)
from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch

# Algorithms (legacy TensorFlow 1.x backend).
# TF1 is not compatible with modern Python/NumPy, so these are only available
# if you separately install a compatible tensorflow build. Without it, the
# PyTorch backend above is used and the tf1 names are simply not exported.
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
    from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
    from spinup.algos.tf1.sac.sac import sac as sac_tf1
    from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
    from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
    from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1
except ImportError:
    pass

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__
