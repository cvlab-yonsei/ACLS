import numpy as np
import torch
from typing import Dict, Optional, List


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeter:
    """A class wrapper to record the values of a loss function.
    Support loss function with mutiple returning terms [num_terms]
    """
    def __init__(self, num_terms: int = 1, names: Optional[List] = None) -> None:
        self.num_terms = num_terms
        self.names = (
            names if names is not None
            else ["loss" if i == 0 else "loss_" + str(i) for i in range(self.num_terms)]
        )
        self.meters = [AverageMeter() for _ in range(self.num_terms)]

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def avg(self, index=None):
        if index is None:
            ret = {}
            for name, meter in zip(self.names, self.meters):
                ret[name] = meter.avg
            return ret
        else:
            return self.meters[index].avg

    def update(self, val, n: int = 1):
        if not isinstance(val, tuple):
            val = [val]
        for x, meter in zip(val, self.meters):
            if isinstance(x, torch.Tensor):
                x = x.item()
            meter.update(x, n)

    def get_vals(self) -> Dict:
        ret = {}
        for name, meter in zip(self.names, self.meters):
            ret[name] = meter.val
        return ret

    def print_status(self) -> str:
        ret = []
        for name, meter in zip(self.names, self.meters):
            ret.append("{} {:.4f} ({:.4f})".format(name, meter.val, meter.avg))
        return "\t".join(ret)

    def get_avgs(self) -> Dict:
        ret = {}
        for name, meter in zip(self.names, self.meters):
            ret[name] = meter.avg
        return ret

    def print_avg(self) -> str:
        ret = []
        for name, meter in zip(self.names, self.meters):
            ret.append("{} {:.4f}".format(name, meter.avg))
        return "\t".join(ret)

def logit_stats(batch):
    #mean = batch.cpu().data.numpy().mean()

    np.argsort(np.max(batch.cpu().data.numpy(), axis=1))
    batch_numpy = batch.cpu().data.numpy()
    maxValuesPos = batch_numpy.argmax(axis=1)
    maxValuesA = batch_numpy.max(axis=1)

    ### TODO: Improve this
    for i in range(len(maxValuesPos)):
        batch_numpy[i][maxValuesPos[i]]=-10

    maxValuesB = batch_numpy.max(axis=1)

    absDif = maxValuesA-maxValuesB
    meanOfMax = absDif.mean()
    absDifMax = np.amax(absDif)

    return meanOfMax, absDifMax


def logit_stats_v2(batch):
    #mean = batch.cpu().data.numpy().mean()

    minValues = batch.cpu().data.numpy().min(axis=1)
    maxValues = batch.cpu().data.numpy().max(axis=1)

    absDif = maxValues-minValues
    absDifMax = np.amax(absDif)
    meanOfMax = absDif.mean()

    allDiff = np.zeros((batch.cpu().data.numpy().shape))

    for i in range(allDiff.shape[1]):
        allDiff[:, i] = maxValues - batch.cpu().data.numpy()[:, i]

    return meanOfMax, absDifMax, absDif, allDiff


def logit_diff(batch):
    minValues = batch.cpu().data.numpy().min(axis=1)
    maxValues = batch.cpu().data.numpy().max(axis=1)

    all_diff = maxValues - minValues
    max_diff = np.amax(all_diff)
    min_diff = np.amin(all_diff)
    mean_diff = all_diff.mean()

    return all_diff, max_diff, min_diff, mean_diff
