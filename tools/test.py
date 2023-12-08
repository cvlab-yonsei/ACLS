import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from calibrate.engine import Tester, OODTester
from calibrate.utils import set_random_seed

logger = logging.getLogger(__name__)

TESTER = {
    "cv": Tester
}


@hydra.main(config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    with open_dict(cfg):
        cfg.work_dir = os.getcwd()
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    set_random_seed(
        cfg.seed if cfg.seed is not None else None,
        deterministic=True if cfg.seed is not None else False
    )

    tester = TESTER[cfg.task](cfg)
    tester.run()

    logger.info("Job complete !\n")


if __name__ == "__main__":
    main()
