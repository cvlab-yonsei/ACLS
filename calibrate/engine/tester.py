import os.path as osp
import time
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from terminaltables.ascii_table import AsciiTable
from typing import Optional
from calibrate.net import ModelWithTemperature
from calibrate.evaluation import (
    AverageMeter, LossMeter, ClassificationEvaluator, CalibrateEvaluator
)
from calibrate.utils import (
    load_train_checkpoint, load_checkpoint, save_checkpoint, round_dict
)
from calibrate.utils.torch_helper import to_numpy, get_lr

logger = logging.getLogger(__name__)


class Tester:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)
        self.build_data_loader()
        self.build_model(self.cfg.test.checkpoint)
        self.build_meter()

    def build_data_loader(self) -> None:
        # data pipeline
        self.test_loader = instantiate(self.cfg.data.object.test)

    def build_model(self, checkpoint: Optional[str] = "") -> None:
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        logger.info("Model initialized")
        self.checkpoint_path = osp.join(
            self.work_dir, "best.pth" if checkpoint == "" else checkpoint
        )
        load_checkpoint(self.checkpoint_path, self.model, self.device)

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        self.evaluator = ClassificationEvaluator(self.num_classes)
        self.calibrate_evaluator = CalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
            device=self.device,
        )

    def reset_meter(self):
        self.batch_time_meter.reset()
        self.evaluator.reset()
        self.calibrate_evaluator.reset()

    @torch.no_grad()
    def eval_epoch(
        self, data_loader,
        phase="Val",
        temp=1.0,
        post_temp=False
    ) -> None:
        self.reset_meter()
        self.model.eval()

        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
            if post_temp:
                outputs = outputs / temp
            # metric
            self.calibrate_evaluator.update(outputs, labels)
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels)
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            end = time.time()
        self.log_eval_epoch_info(phase)
        if self.cfg.test.save_logits:
            logits_save_path = (
                osp.splitext(self.checkpoint_path)[0]
                + "_logits"
                + ("_pt.npz" if post_temp else ".npz")
            )
            self.calibrate_evaluator.save_npz(logits_save_path)

    def log_eval_epoch_info(self, phase="Val"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        classify_metric, classify_table_data = self.evaluator.mean_score(print=False)
        log_dict.update(classify_metric)
        calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(print=False)
        log_dict.update(calibrate_metric)
        logger.info("{} Epoch\t{}".format(
            phase, json.dumps(round_dict(log_dict))
        ))
        logger.info("\n" + AsciiTable(classify_table_data).table)
        logger.info("\n" + AsciiTable(calibrate_table_data).table)

    def post_temperature(self):
        _, self.val_loader = instantiate(self.cfg.data.object.trainval)
        model_with_temp = ModelWithTemperature(self.model, device=self.device)
        model_with_temp.set_temperature(self.val_loader)
        temp = model_with_temp.get_temperature()
        return temp

    def test(self):
        logger.info(
            "Everything is perfect so far. Let's start testing. Good luck!"
        )
        self.eval_epoch(self.test_loader, phase="Test")
        if self.cfg.test.post_temperature:
            logger.info("Test with post-temperature scaling!")
            temp = self.post_temperature()
            self.eval_epoch(self.test_loader, phase="TestPT", temp=temp, post_temp=True)

    def run(self):
        self.test()
