'''
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
'''
import logging
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from calibrate.evaluation.metrics import ECELoss

logger = logging.getLogger(__name__)


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device="cuda:0", log=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.log = log
        self.device = device

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature

    def set_temperature_ng(self, embedding_model, x_val, y_val,
                           cross_validate="ece",
                           batch_size=128):
        self.model.eval()
        embedding_model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        batch_size = 128
        with torch.no_grad():
            for i in range(1 + x_val.shape[0]//batch_size):
                data = torch.from_numpy(
                    x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]
                ).type(torch.LongTensor).to(self.device)
                labels = torch.from_numpy(
                    np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)
                ).to(self.device)
                emb = embedding_model(data)
                logits = self.model(emb)
                logits_list.append(logits)
                labels_list.append(labels)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            logger.info(
                'Before temperature - NLL: {:.4f}, ECE: {:.4f}'.format(
                    before_temperature_nll, before_temperature_ece
                )
            )

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            logger.info(
                'Optimal temperature: {:.3f}'.format(self.temperature)
            )
            logger.info(
                'After temperature - NLL: {:.4f}, ECE: {:.4f}'.format(
                    after_temperature_nll, after_temperature_ece
                )
            )

    def set_temperature(self,
                        valid_loader,
                        cross_validate='ece'):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        # self.cuda()
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            logger.info(
                'Before temperature - NLL: {:.4f}, ECE: {:.4f}'.format(
                    before_temperature_nll, before_temperature_ece
                )
            )

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            logger.info(
                'Optimal temperature: {:.3f}'.format(self.temperature)
            )
            logger.info(
                'After temperature - NLL: {:.4f}, ECE: {:.4f}'.format(
                    after_temperature_nll, after_temperature_ece
                )
            )

    def get_temperature(self):
        return self.temperature
