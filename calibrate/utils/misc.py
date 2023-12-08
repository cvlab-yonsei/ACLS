import logging
import numpy as np
import os
import random
from datetime import datetime
import torch
from copy import deepcopy

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = None, deterministic: bool = False):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
        deterministic (bool):  Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logfile(logger):
    if len(logger.root.handlers) == 1:
        return None
    else:
        return logger.root.handlers[1].baseFilename


def round_dict(d, decimals=5):
    """
    Return a new dictionary with all the flating values rounded
    with the sepcified number of decimals
    """
    ret = deepcopy(d)
    for key in ret:
        if isinstance(ret[key], float):
            ret[key] = round(ret[key], decimals)
    return ret
