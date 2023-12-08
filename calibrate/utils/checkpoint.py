import os
import os.path as osp
import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_checkpoint(
    model_path: str,
    model: torch.nn.Module,
    device: torch.device
) -> None:
    if not osp.exists(model_path):
        raise FileNotFoundError(
            "Model not found : {}".format(model_path)
        )
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    checkpoint = dict(
        (key[7:] if "module" in key else key, value)
        for (key, value) in checkpoint.items()
    )
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    logger.info("Succeed to load weights from {}".format(model_path))
    if missing_keys:
        logger.warn("Missing keys : {}".format(missing_keys))
    if unexpected_keys:
        logger.warn("Unexpected keys : {}".format(unexpected_keys))


def load_train_checkpoint(
    work_dir: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None
) -> Tuple:

    try:
        last_checkpoint_path = osp.join(work_dir, "last.pth")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("Succeed to load train info from {}".format(last_checkpoint_path))

        best_checkpoint_path = osp.join(work_dir, "best.pth")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        best_epoch = checkpoint["epoch"]
        best_score = checkpoint["val_score"] if "val_score" in checkpoint else None
        return epoch + 1, best_epoch, best_score
    except Exception:
        return 0, -1, None


def save_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epoch: int,
    best_checkpoint: bool = False,
    val_score: Optional[float] = None,
    keep_checkpoint_num: int = 1,
    keep_checkpoint_interval: int = 0
) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None
    }
    if val_score:
        state["val_score"] = val_score
    torch.save(state, osp.join(save_dir, "last.pth"))
    if best_checkpoint:
        torch.save(state, osp.join(save_dir, "best.pth"))
    if keep_checkpoint_num > 1:
        torch.save(state, osp.join(save_dir, "epoch_{}.pth".format(epoch + 1)))
        remove_file = osp.join(save_dir, "epoch_{}.pth".format(epoch + 1 - keep_checkpoint_num))
        if osp.exists(remove_file):
            os.remove(remove_file)
    if keep_checkpoint_interval > 0:
        if (epoch + 1) % keep_checkpoint_interval == 0:
            torch.save(
                state, osp.join(save_dir, "epoch_{}.pth".format(epoch + 1))
            )
