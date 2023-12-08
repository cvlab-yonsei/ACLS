import random
import numpy as np
import torch
import torch.nn as nn

from .constants import EPS


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param in optimizer.param_groups:
        return param["lr"]


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def kl_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """kl divergence of two distribution:
        KL(p||q) = \sum p log(p/q)
    """
    p = torch.flatten(p, 1)
    q = torch.flatten(q, 1)

    y = (p * torch.log(p / (q + EPS) + EPS)).mean()

    return y


def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of input distribution:
        EN(x) = - \sum x * log(x)
    """
    x = torch.flatten(x, 1)
    y = - x * torch.log(x + EPS)
    y = y.mean()

    return y


def disable_bn(model):
    for module in model.modules():
        if isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ):
            module.eval()


def enable_bn(model):
    model.train()
