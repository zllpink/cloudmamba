import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from torchmetrics.classification import Accuracy, JaccardIndex, Recall
from torchmetrics.classification import Precision, Specificity, F1Score, CohenKappa

from utils.cloud_dection import ImageDataset
from .trainers import data_prefetcher


def apply_color_map(num_classes, imgs):
    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        cmap[i] = np.array([(i / (num_classes - 1)) * 255] * 3)

    out = []
    imgs = imgs.cpu().numpy()
    for i in range(len(imgs)):
        img = imgs[i]
        img = Image.fromarray(img.astype(np.uint8), mode="P")
        img.putpalette(cmap)
        img = img.convert('RGB')
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]
        out.append(img)
    out = np.concatenate(out, axis=0)
    return torch.from_numpy(out)


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


def norm_range(t, value_range=None):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
    t.mul_(255).add_(0.5).clamp_(0, 255)


def sample_images(args, dataloader, model, epoch):
    """Saves a generated sample from the validation set"""
    with torch.no_grad():
        image, label = next(dataloader)
        predict = get_pred(args, model, image)
        predict = torch.argmax(predict, dim=1, keepdim=True)

        label_vis   = apply_color_map(args.num_classes, label).to(image.device)
        predict_vis = apply_color_map(args.num_classes, predict.squeeze()).to(image.device)
        norm_range(image)

        # show only first 3 channels for RGB display if in_channels >= 3
        disp = image[:, :3, ...] if image.size(1) >= 3 else image.expand(-1, 3, -1, -1)

        img_sample = torch.cat((disp, predict_vis, label_vis), dim=-2)
        save_image(
            img_sample / 255,
            os.path.join("checkpoints", args.save_name, "show", str(epoch) + '.png'),
            nrow=args.batch_size,
            normalize=False
        )


class Evaluator(object):

    def __init__(self, num_classes, device, average='macro') -> None:
        self.device = device
        self.indicators = ['Accuracy', 'JaccardIndex', 'Recall', 'Precision',
                           'Specificity', 'F1Score', 'CohenKappa']
        task = 'binary' if num_classes == 2 else 'multiclass'

        self.ac    = Accuracy(task=task, average=average, num_classes=num_classes).to(device)
        self.ji    = JaccardIndex(task=task, average=average, num_classes=num_classes).to(device)
        self.rc    = Recall(task=task, average=average, num_classes=num_classes).to(device)
        self.pc    = Precision(task=task, average=average, num_classes=num_classes).to(device)
        self.sp    = Specificity(task=task, average=average, num_classes=num_classes).to(device)
        self.f1    = F1Score(task=task, average=average, num_classes=num_classes).to(device)
        self.kappa = CohenKappa(task=task, num_classes=num_classes).to(device)
        self.funcs   = [self.ac, self.ji, self.rc, self.pc, self.sp, self.f1, self.kappa]
        self.metrics = torch.zeros(len(self.funcs), dtype=torch.float64).to(device)
        self.num = 0

    def send(self, pred, target):
        for i, func in enumerate(self.funcs):
            value = func(pred, target)
            if torch.isnan(value):
                self.metrics[i] += self.metrics[i] / max(self.num, 1)
            else:
                self.metrics[i] += value
        self.num += 1

    def result(self):
        return self.metrics.cpu().numpy() / max(self.num, 1)


def get_pred(args, model, image):
    if args.model_name == 'cdnetv2':
        predict, _ = model(image)
    else:
        predict = model(image)
    return predict


def save_results(evl_path, metrics, indicators, epoch):
    try:
        metrics_df = pd.read_excel(evl_path, sheet_name='Sheet1')
    except Exception:
        metrics_df = pd.DataFrame(columns=['Epoch'] + indicators)
    row = metrics_df.shape[0]
    metrics_df.loc[row, 'Epoch'] = epoch
    metrics_df.loc[row, indicators] = metrics

    try:
        if os.path.exists(evl_path):
            ew = pd.ExcelWriter(evl_path, mode='a', if_sheet_exists='replace', engine='openpyxl')
        else:
            ew = pd.ExcelWriter(evl_path)
        metrics_df.to_excel(ew, index=False, sheet_name='Sheet1')
        ew.close()
    except Exception as e:
        print(e)


def evaluate(args, model, device, epoch):
    test_dataset = ImageDataset(args, mode="val", normalization=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_cpu,
        pin_memory=True
    )
    evaluator = Evaluator(args.num_classes, device, average='macro')

    with torch.no_grad():
        prefetcher = data_prefetcher(test_loader)
        image, label = prefetcher.next()
        while image is not None:
            predict = get_pred(args, model, image)
            predict = torch.argmax(predict, dim=1).to(device, dtype=torch.long)
            evaluator.send(predict, label)
            image, label = prefetcher.next()

    metrics = evaluator.result()
    evl_path = os.path.join(
        "checkpoints", args.save_name, "evaluation", f'{args.time}-macro.xlsx'
    )
    save_results(evl_path, metrics, evaluator.indicators, epoch)
    return metrics[0]   # Accuracy
