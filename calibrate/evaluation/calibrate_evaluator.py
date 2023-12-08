import logging
from terminaltables import AsciiTable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from .reliability_diagram import ReliabilityDiagram
from calibrate.utils.torch_helper import to_numpy

logger = logging.getLogger(__name__)


class CalibrateEvaluator(DatasetEvaluator):
    def __init__(self, num_classes, num_bins=15, device="cuda:0") -> None:
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.logits = None
        self.labels = None

    def num_samples(self):
        return (
            self.labels.shape[0]
            if self.labels is not None
            else 0
        )

    def main_metric(self) -> None:
        return "ece"

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """update

        Args:
            logits (torch.Tensor): n x num_classes
            label (torch.Tensor): n x 1
        """
        assert logits.shape[0] == labels.shape[0]
        if self.logits is None:
            self.logits = logits
            self.labels = labels
        else:
            self.logits = torch.cat((self.logits, logits), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def mean_score(self, print=False, all_metric=True):
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss(self.num_bins).to(self.device)
        aece_criterion = AdaptiveECELoss(self.num_bins).to(self.device)
        cece_criterion = ClasswiseECELoss(self.num_bins).to(self.device)

        nll = nll_criterion(self.logits, self.labels).item()
        ece = ece_criterion(self.logits, self.labels).item()
        aece = aece_criterion(self.logits, self.labels).item()
        cece = cece_criterion(self.logits, self.labels).item()

        metric = {"nll": nll, "ece": ece, "aece": aece, "cece": cece}

        columns = ["samples", "nll", "ece", "aece", "cece"]
        table_data = [columns]
        table_data.append(
            [
                self.num_samples(),
                "{:.5f}".format(nll),
                "{:.5f}".format(ece),
                "{:.5f}".format(aece),
                "{:.5f}".format(cece),
            ]
        )

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        if all_metric:
            return metric, table_data
        else:
            return metric[self.main_metric()]

    def plot_reliability_diagram(self, title=""):
        diagram = ReliabilityDiagram(bins=25, style="curve")
        probs = F.softmax(self.logits, dim=1)
        fig_reliab, fig_hist = diagram.plot(
            to_numpy(probs), to_numpy(self.labels),
            title_suffix=title
        )
        return fig_reliab, fig_hist

    def save_npz(self, save_path):
        np.savez(
            save_path,
            logits=to_numpy(self.logits),
            labels=to_numpy(self.labels)
        )
