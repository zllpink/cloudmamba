import sys
import time
import datetime

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from .loss import SimpleCloudCELoss


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_image, self.next_label = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image, label = self.next_image, self.next_label
        if image is not None:
            image.record_stream(torch.cuda.current_stream())
        if label is not None:
            label.record_stream(torch.cuda.current_stream())
        self.preload()
        return image, label


class BaseTrainer(object):

    def __init__(self, args, model, device) -> None:
        self.model_name = args.model_name
        self.model = model
        self.device = device
        self.args = args
        self.criterion = SimpleCloudCELoss().to(device)
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer, self.lr_scheduler = None, None
        lr_decay_fn = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
        if self.model_name == 'cdnetv2':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=1e-4,
                momentum=0.9, weight_decay=0.0005
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(self.args.b1, self.args.b2)
            )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_decay_fn
        )
        if self.args.start_epoch > 1:
            self.lr_scheduler.step(self.args.start_epoch - 1)

    def cal_loss(self, image, label):
        if self.model_name == 'cdnetv2':
            pred, pred_aux = self.model(image)
            loss = self.criterion(pred, label) + self.criterion(pred_aux, label)
        else:
            pred = self.model(image)
            loss = self.criterion(pred, label)
        return loss

    def train(self, epoch, train_loader):
        prev_time = time.time()
        prefetcher = data_prefetcher(train_loader)
        image, label = prefetcher.next()
        i = 1
        epoch_loss = 0.0
        num_batches = 0
        while image is not None:
            self.optimizer.zero_grad()
            loss = self.cal_loss(image, label)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            batches_done = (epoch - 1) * len(train_loader) + i
            batches_left = self.args.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=int(batches_left * (time.time() - prev_time))
            )
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %03d/%d] [Batch %03d/%d] [CE Loss: %7.4f] ETA: %8s"
                % (epoch, self.args.n_epochs, i, len(train_loader),
                   loss.item(), time_left)
            )
            i += 1
            image, label = prefetcher.next()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if num_batches == 0:
            return 0.0
        return epoch_loss / num_batches
