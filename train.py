import os
import sys
import argparse

# Parse --gpu and set CUDA_VISIBLE_DEVICES before importing torch,
# since PyTorch scans GPUs on first import.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=str, default="0")
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import time
import torch
from torch.utils.data import DataLoader

from config import Options
from models.model_zoo import get_model
from utils.trainers import BaseTrainer
from utils.metric import evaluate, sample_images
from utils.cloud_dection import ImageDataset, ForeverDataIterator


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cudnn.enabled = False


def _ckpt_dir(args):
    return os.path.join("checkpoints", args.save_name, "saved_models")


def save_model(model, args, name, trainer=None):
    """Save model weights; also save optimizer and lr_scheduler states if trainer is provided."""
    payload = {'model': model.state_dict()}
    if trainer is not None:
        payload['optimizer'] = trainer.optimizer.state_dict()
        if trainer.lr_scheduler is not None:
            payload['lr_scheduler'] = trainer.lr_scheduler.state_dict()
    os.makedirs(_ckpt_dir(args), exist_ok=True)
    torch.save(payload, os.path.join(_ckpt_dir(args), f"{args.time}_{name}.pth"))


def _remove_old_best(args, tag):
    """Delete old checkpoints with the same tag (best_loss / best_acc) to avoid disk accumulation."""
    prefix = f"{args.time}_{tag}_"
    for fname in os.listdir(_ckpt_dir(args)):
        if fname.startswith(prefix) and fname.endswith(".pth"):
            try:
                os.remove(os.path.join(_ckpt_dir(args), fname))
            except OSError:
                pass


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MODEL
    model = get_model(args, device)

    if args.checkpoint != '0':
        # Load pretrained models
        checkpoint = torch.load(os.path.join(_ckpt_dir(args), f"{args.checkpoint}.pth"))
        model.load_state_dict(checkpoint['model'])

    trainer = BaseTrainer(args, model, device)

    # Restore optimizer and scheduler states for resume training
    if args.checkpoint != '0' and 'optimizer' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    if args.checkpoint != '0' and 'lr_scheduler' in checkpoint and trainer.lr_scheduler is not None:
        trainer.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # Configure dataloaders
    train_loader = DataLoader(
        ImageDataset(args, mode="train", normalization=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True
    )
    test_loader = DataLoader(
        ImageDataset(args, mode="val", normalization=True),
        batch_size=args.batch_size,
        num_workers=1,
        drop_last=True
    )
    test_loader = ForeverDataIterator(test_loader, device)

    #  Training
    best_loss = float("inf")
    best_acc  = 0.0
    for epoch in range(args.start_epoch, args.n_epochs+1):
        model.train()
        epoch_loss = trainer.train(epoch, train_loader)
        model.eval()

        # If at sample interval save image
        if args.sample_interval and epoch % args.sample_interval == 0:
            sample_images(args, test_loader, model, epoch)

        # Evaluate and print accuracy and loss every epoch
        acc_ = evaluate(args, model, device, epoch)
        print(f"\nEpoch {epoch:03d}/{args.n_epochs} - Loss: {epoch_loss:.4f} - Acc: {acc_:.4f}")

        # Save best_loss model (auto-delete old file, keep only latest)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _remove_old_best(args, "best_loss")
            save_model(model, args, name=f"best_loss_epoch_{epoch:03d}")
            print(f"  -> best_loss updated: {best_loss:.4f}")

        # Save best_acc model (auto-delete old file, keep only latest)
        if acc_ > best_acc:
            best_acc = acc_
            _remove_old_best(args, "best_acc")
            save_model(model, args, name=f"best_acc_epoch_{epoch:03d}")
            print(f"  -> best_acc  updated: {best_acc:.4f}")

        # Periodic checkpoint (with optimizer state for resume training)
        if epoch > 1 and args.checkpoint_interval and epoch % args.checkpoint_interval == 0:
            save_model(model, args, name=f"ckpt_epoch_{epoch:03d}", trainer=trainer)


if __name__ == '__main__':
    # time.sleep(2 * 60 * 60) # python train.py --gpu 0
    args = Options(model_name='hrcloudnet').parse(save_args=True)  # "cloudnet", "cdnetv2", "hrcloudnet", "mscff", "swinunet", "rdunet", "cloudmamba"
    main(args)
