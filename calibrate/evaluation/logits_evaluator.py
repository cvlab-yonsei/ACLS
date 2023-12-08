import numpy as np

from .evaluator import DatasetEvaluator


class LogitsEvaluator(DatasetEvaluator):
    """get logit differences
    mean_diff : (max value of logits - value of logits).mean() 
    max_diff : (max value of logits - value of logits).max()
    margin : max value of logits - second max value of logits

    Args:
        DatasetEvaluator ([type]): [description]
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean_diffs = []
        self.max_diffs = []
        self.margins = []

    def num_samples(self):
        return self.count

    def main_metric(self):
        return "mean_diffs"

    def update(self, logits: np.ndarray):
        n = logits.shape[0]
        self.count += n
        sort_inds = np.argsort(logits, axis=1)
        max_values = np.zeros(n)
        second_max_values = np.zeros(n)
        min_values = np.zeros(n)
        for i in range(n):
            max_values[i] = logits[i, sort_inds[i, -1]]
            second_max_values[i] = logits[i, sort_inds[i, -2]]
            min_values[i] = logits[i, sort_inds[i, 0]]
        # max_values = logits[:, sort_inds[:, -1]]
        # second_max_values = logits[:, sort_inds[:, -2]]

        diffs = np.repeat(max_values.reshape(n, 1), logits.shape[1], axis=1) - logits
        # self.mean_diffs.append(diffs.sum())
        self.mean_diffs.append(np.sum(diffs, axis=1) / (logits.shape[1] - 1))
        self.max_diffs.append(np.max(diffs, axis=1))

        margins = max_values - second_max_values
        self.margins.append(margins)

        return np.mean(self.mean_diffs[-1])

    def curr_score(self):
        return {
            self.main_metric(): np.mean(self.mean_diffs[-1])
        }

    def mean_score(self, all_metric=True):
        mean_diffs = np.concatenate(self.mean_diffs)
        max_diffs = np.concatenate(self.max_diffs)
        margins = np.concatenate(self.margins)

        if not all_metric:
            return np.mean(self.mean_diffs)

        metric = {}
        metric["mean_diffs"] = np.mean(mean_diffs)
        metric["max_diffs"] = np.mean(max_diffs)
        metric["margin"] = np.mean(margins)

        return metric
