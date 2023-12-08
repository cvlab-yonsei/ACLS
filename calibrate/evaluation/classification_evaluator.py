import logging
from terminaltables import AsciiTable
import numpy as np
from sklearn.metrics import top_k_accuracy_score, confusion_matrix

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class ClassificationEvaluator(DatasetEvaluator):
    def __init__(
        self,
        num_classes: int
    ) -> None:
        self.num_classes = num_classes

    def reset(self) -> None:
        self.preds = None
        self.labels = None

    def main_metric(self) -> None:
        return "acc"

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def update(self, pred: np.ndarray, label: np.ndarray) -> float:
        """update

        Args:
            pred (np.ndarray): n x num_classes
            label (np.ndarray): n x 1

        Returns:
            float: acc
        """
        assert pred.shape[0] == label.shape[0]
        if self.preds is None:
            self.preds = pred
            self.labels = label
        else:
            self.preds = np.concatenate((self.preds, pred), axis=0)
            self.labels = np.concatenate((self.labels, label), axis=0)

        pred_label = np.argmax(pred, axis=1)
        acc = (pred_label == label).astype("int").sum() / label.shape[0]

        # acc = top_k_accuracy_score(label, pred, k=1)

        self.curr = {"acc": acc}
        return acc

    def curr_score(self):
        return self.curr

    def mean_score(self, print=False, all_metric=True):
        # acc = (
        #     (self.preds == self.labels).astype("int").sum()
        #     / self.labels.shape[0]
        # )
        acc = top_k_accuracy_score(self.labels, self.preds, k=1)
        acc_5 = top_k_accuracy_score(self.labels, self.preds, k=5)

        pred_labels = np.argmax(self.preds, axis=1)
        confusion = confusion_matrix(self.labels, pred_labels, normalize="true")
        macc = np.diagonal(confusion).mean()

        metric = {"acc": acc, "acc_5": acc_5, "macc": macc}

        columns = ["samples", "acc", "acc_5", "macc"]
        table_data = [columns]
        table_data.append(
            [
                self.num_samples(),
                "{:.5f}".format(acc),
                "{:.5f}".format(acc_5),
                "{:.5f}".format(macc)
            ]
        )

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        if all_metric:
            return metric, table_data
        else:
            return metric[self.main_metric()], table_data

