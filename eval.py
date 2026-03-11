#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import torch
import numpy as np
import cv2
import tifffile
import argparse
import time
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix, jaccard_score
)

from models.model_zoo import get_model


# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────

# Binary class names and pixel value mapping (38-Cloud)
CLASS_NAMES  = ['clear', 'cloud']
LABEL_TO_PIX = {0: 0, 1: 255}    # class index -> pixel value for saving
PIX_TO_LABEL = {0: 0, 255: 1}    # pixel value -> class index


_MAX_VAL = 4500.0   # fixed normalization upper bound, consistent with training


def read_image(path: str, in_channels: int) -> np.ndarray:
    """Read an image (multi-band TIF / JPG / PNG) and normalize to [0, 1]
    by dividing by _MAX_VAL, consistent with training preprocessing.
    Returns float32 [H, W, in_channels].

    Layout detection:
      - (H, W)      -> expanded to (H, W, 1)
      - (C, H, W)   -> transposed to (H, W, C) when dim-0 <= 16 and much smaller than spatial dims
      - (H, W, C)   -> kept as-is
    """
    try:
        img = tifffile.imread(path)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")

    if img.ndim == 2:
        img = img[:, :, np.newaxis]                           # [H, W] -> [H, W, 1]
    elif img.ndim == 3:
        # CHW detection: dim-0 is channel count (<= 16) and much smaller than spatial dims
        if img.shape[0] <= 16 and img.shape[0] < img.shape[1] // 2:
            img = img.transpose(1, 2, 0)                      # [C, H, W] -> [H, W, C]
        # otherwise treat as HWC, no transpose

    c = img.shape[2]
    # align channel count
    if c < in_channels:
        repeats = in_channels // c
        remain  = in_channels  % c
        parts   = [img] * repeats + ([img[:, :, :remain]] if remain else [])
        img = np.concatenate(parts, axis=2)
    elif c > in_channels:
        img = img[:, :, :in_channels]

    # divide by 4500, consistent with training ImageDataset
    img = img.astype(np.float32) / _MAX_VAL
    img = np.clip(img, 0.0, 1.0)
    return img


def read_label(path: str) -> np.ndarray:
    """Read a binary label map, returns int64 [H, W] with cloud=1, clear=0.
    Compatible with air-cd (values 0/1) and 38-Cloud (values 0/255).
    """
    try:
        lbl = tifffile.imread(path)
    except Exception:
        lbl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if lbl is None:
        raise FileNotFoundError(f"Cannot read label: {path}")
    if lbl.ndim == 3:
        lbl = lbl[0] if lbl.shape[0] <= 4 else lbl[:, :, 0]
    out = np.zeros_like(lbl, dtype=np.int64)
    out[lbl > 0] = 1   # compatible with air-cd (0/1) and 38-Cloud (0/255)
    return out


def save_pred(pred: np.ndarray, path: str):
    """Save prediction: class 0->0 (clear), class 1->255 (cloud)."""
    vis = np.zeros_like(pred, dtype=np.uint8)
    vis[pred == 1] = 255
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, vis)


def metrics_from_cm(cm: np.ndarray, file=None):
    """Compute and print all metrics directly from confusion matrix (fast, no pixel arrays)."""
    num_classes = cm.shape[0]
    names  = CLASS_NAMES[:num_classes]
    total  = cm.sum()
    eps    = 1e-8

    # OA
    oa = cm.diagonal().sum() / (total + eps)

    # per-class P / R / F1 / IoU
    p_per, r_per, f1_per, iou_per = [], [], [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        pi  = tp / (tp + fp + eps)
        ri  = tp / (tp + fn + eps)
        fi  = 2 * pi * ri / (pi + ri + eps)
        iou = tp / (tp + fp + fn + eps)
        p_per.append(pi); r_per.append(ri)
        f1_per.append(fi); iou_per.append(iou)

    p_macro  = np.mean(p_per)
    r_macro  = np.mean(r_per)
    f1_macro = np.mean(f1_per)
    f1_micro = oa                       # micro F1 == OA for multiclass
    miou     = np.mean(iou_per)

    # Kappa
    pe = sum((cm[i, :].sum() * cm[:, i].sum()) for i in range(num_classes))
    pe /= (total ** 2 + eps)
    kappa = (oa - pe) / (1 - pe + eps)

    def out(s=''):
        print(s)
        if file:
            file.write(s + '\n')

    out(f"\n  Main Metrics:")
    out(f"    Overall Accuracy (OA) : {oa:.4f}  ({oa*100:.2f}%)")
    out(f"    mIoU (Jaccard)        : {miou:.4f}  ({miou*100:.2f}%)")
    out(f"    Kappa Coefficient     : {kappa:.4f}")
    out(f"    F1  (macro)           : {f1_macro:.4f}  ({f1_macro*100:.2f}%)")
    out(f"    F1  (micro)           : {f1_micro:.4f}  ({f1_micro*100:.2f}%)")
    out(f"\n  Precision / Recall (macro):")
    out(f"    Precision             : {p_macro:.4f}  ({p_macro*100:.2f}%)")
    out(f"    Recall                : {r_macro:.4f}  ({r_macro*100:.2f}%)")
    out(f"\n  Per-class Metrics:")
    out(f"    {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
    out("    " + "-" * 52)
    for i, name in enumerate(names):
        out(f"    {name:<10} {p_per[i]:>10.4f} {r_per[i]:>10.4f} "
            f"{f1_per[i]:>10.4f} {iou_per[i]:>10.4f}")
    out(f"    {'Overall':<10} {p_macro:>10.4f} {r_macro:>10.4f} "
        f"{f1_macro:>10.4f} {miou:>10.4f}  <- macro mean  |  OA={oa:.4f}  Kappa={kappa:.4f}")
    out(f"\n  Confusion Matrix  (rows=GT, cols=Pred):")
    out("    GT\\Pred" + "".join(f"{n:>10}" for n in names))
    for i, name in enumerate(names):
        out(f"    {name:<8}" + "".join(f"{int(cm[i,j]):>10}" for j in range(num_classes)))

    return dict(OA=oa, mIoU=miou, Kappa=kappa,
                F1_macro=f1_macro, P_macro=p_macro, R_macro=r_macro,
                iou_per=iou_per)


def print_metrics(yt: np.ndarray, yp: np.ndarray,
                  num_classes: int = 3, file=None):
    """Compute full evaluation metrics from raw pixel arrays (single image use)."""
    cm = confusion_matrix(yt, yp, labels=list(range(num_classes)))
    return metrics_from_cm(cm, file=file)


# ────────────────────────────────────────────────────────────
# Core inference: sliding window (probability accumulation)
# ────────────────────────────────────────────────────────────

def sliding_predict(model, img_np: np.ndarray, device,
                    num_classes: int = 3,
                    patch_size: int = 512,
                    overlap: int = 64) -> np.ndarray:
    """Sliding-window inference: accumulate softmax probabilities, then argmax.
    img_np : float32 [H, W, C], value range [0, 1]
    returns: int64  [H, W], class indices 0/1/...
    """
    import torchvision.transforms.functional as TF

    H, W, C = img_np.shape
    stride = patch_size - overlap

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h or pad_w:
        img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    H_pad, W_pad = img_np.shape[:2]
    prob_acc = np.zeros((num_classes, H_pad, W_pad), dtype=np.float32)
    count    = np.zeros((H_pad, W_pad), dtype=np.float32)

    img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
    img_t = TF.normalize(img_t, [0.5] * C, [0.5] * C)

    rows = list(range(0, H_pad - patch_size + 1, stride))
    cols = list(range(0, W_pad - patch_size + 1, stride))

    model.eval()
    with torch.no_grad():
        for y in rows:
            for x in cols:
                patch = img_t[:, y:y+patch_size, x:x+patch_size].unsqueeze(0).to(device)
                out = model(patch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]   # [C, ps, ps]
                prob_acc[:, y:y+patch_size, x:x+patch_size] += prob
                count[y:y+patch_size, x:x+patch_size] += 1

    prob_acc /= np.maximum(count[np.newaxis], 1e-6)
    pred = np.argmax(prob_acc, axis=0).astype(np.int64)
    return pred[:H, :W]


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def predict_single(model, image_path: str, output_path: str, device,
                   num_classes: int = 3, in_channels: int = 4,
                   patch_size: int = 512, overlap: int = 64) -> np.ndarray:
    """Run inference on a single image, save result, return prediction map [H, W]."""
    img_np = read_image(image_path, in_channels)
    H, W   = img_np.shape[:2]
    print(f"  Image size: {H} x {W} x {in_channels}")

    pred = sliding_predict(model, img_np, device, num_classes, patch_size, overlap)
    save_pred(pred, output_path)
    print(f"  Saved -> {output_path}")
    return pred


def predict_batch(model, image_dir: str, output_dir: str, device,
                  num_classes: int = 3, in_channels: int = 4,
                  patch_size: int = 512, overlap: int = 64,
                  exts=('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
    """Batch inference (no labels), save prediction maps only."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in exts])
    if not files:
        print(f"No image files found in {image_dir}"); return

    total = len(files)
    print(f"Found {total} images, starting inference ...")
    ok, fail = 0, []

    for idx, fname in enumerate(files, 1):
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{stem}_pred.png")
        print(f"\n[{idx}/{total}] {fname}")
        try:
            predict_single(model, os.path.join(image_dir, fname), out_path,
                           device, num_classes, in_channels, patch_size, overlap)
            ok += 1
        except Exception as e:
            print(f"  Failed: {e}"); fail.append(fname)

    print(f"\nDone  success={ok}  failed={len(fail)}")
    if fail:
        print("Failed files:", "\n  ".join(fail))


def predict_batch_with_metrics(model, image_dir: str, label_dir: str,
                               output_dir: str, device,
                               num_classes: int = 3, in_channels: int = 4,
                               patch_size: int = 512, overlap: int = 64,
                               label_exts=('.tif', '.tiff', '.png', '.TIF', '.TIFF', '.PNG'),
                               img_exts=('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
    """Batch inference + accuracy evaluation (per-image metrics + summary + confusion matrix)."""
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in img_exts])
    if not files:
        print(f"No image files found in {image_dir}"); return

    total = len(files)
    print(f"Found {total} images, starting inference + evaluation ...")

    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("WHUS2 Binary Cloud Detection Evaluation Report\n" + "="*60 + "\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Num classes: {num_classes}  Classes: {CLASS_NAMES[:num_classes]}\n")
        f.write(f"Test images: {total}\n" + "="*60 + "\n\n")

    cum_cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # accumulated confusion matrix
    ok, fail = 0, []

    for idx, fname in enumerate(files, 1):
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        out_path   = os.path.join(output_dir, f"{stem}_pred.png")

        # search for label file with multiple stem variants
        # supports 38-Cloud test set prefix "edited_corrected_gts_"
        import re
        stem_candidates = [stem,
                           f"edited_corrected_gts_{stem}",
                           re.sub(r'_\d+$', '', stem)]

        lbl_path = None
        for s in stem_candidates:
            for ext in label_exts:
                cand = os.path.join(label_dir, s + ext)
                if os.path.exists(cand):
                    lbl_path = cand; break
            if lbl_path:
                break
        if lbl_path is None:
            print(f"\n[{idx}/{total}] {fname}  - label not found, skipped")
            fail.append(fname); continue

        print(f"\n[{idx}/{total}] {fname}")
        t0 = time.time()

        try:
            pred  = predict_single(model, image_path, out_path, device,
                                   num_classes, in_channels, patch_size, overlap)
            label = read_label(lbl_path)

            # align spatial dimensions
            if pred.shape != label.shape:
                label = cv2.resize(label.astype(np.uint8),
                                   (pred.shape[1], pred.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int64)

            yt = label.flatten().astype(np.int32)
            yp = pred.flatten().astype(np.int32)

            # per-image metrics
            oa_s   = accuracy_score(yt, yp)
            k_s    = cohen_kappa_score(yt, yp)
            miou_s = jaccard_score(yt, yp, average='macro', zero_division=0)
            f1_s   = f1_score(yt, yp,        average='macro', zero_division=0)
            p_s    = precision_score(yt, yp,  average='macro', zero_division=0)
            r_s    = recall_score(yt, yp,     average='macro', zero_division=0)
            p_per  = precision_score(yt, yp, average=None, zero_division=0)
            r_per  = recall_score(yt, yp,    average=None, zero_division=0)
            f1_per = f1_score(yt, yp,        average=None, zero_division=0)
            cm_s   = confusion_matrix(yt, yp, labels=list(range(num_classes)))
            elapsed = time.time() - t0

            names = CLASS_NAMES[:num_classes]
            # per-class IoU
            iou_per = []
            for c in range(num_classes):
                tp = cm_s[c, c]
                iou_c = tp / (cm_s[c, :].sum() + cm_s[:, c].sum() - tp + 1e-8)
                iou_per.append(iou_c)

            # print
            print(f"  OA={oa_s*100:.2f}%  mIoU={miou_s*100:.2f}%  "
                  f"F1={f1_s*100:.2f}%  P={p_s*100:.2f}%  "
                  f"R={r_s*100:.2f}%  Kappa={k_s:.4f}  ({elapsed:.1f}s)")
            print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
            print("  " + "-" * 50)
            for c, name in enumerate(names):
                pi  = p_per[c]  if c < len(p_per)  else 0.0
                ri  = r_per[c]  if c < len(r_per)  else 0.0
                fi  = f1_per[c] if c < len(f1_per) else 0.0
                print(f"  {name:<10} {pi*100:>9.2f}% {ri*100:>9.2f}% "
                      f"{fi*100:>9.2f}% {iou_per[c]*100:>9.2f}%")

            # write to report
            lines = [f"[{idx}] {fname}\n",
                     f"  OA={oa_s*100:.2f}%  mIoU={miou_s*100:.2f}%  "
                     f"F1={f1_s*100:.2f}%  P={p_s*100:.2f}%  "
                     f"R={r_s*100:.2f}%  Kappa={k_s:.4f}\n",
                     f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}\n"]
            for c, name in enumerate(names):
                pi  = p_per[c]  if c < len(p_per)  else 0.0
                ri  = r_per[c]  if c < len(r_per)  else 0.0
                fi  = f1_per[c] if c < len(f1_per) else 0.0
                lines.append(f"  {name:<10} {pi*100:>9.2f}% {ri*100:>9.2f}% "
                              f"{fi*100:>9.2f}% {iou_per[c]*100:>9.2f}%\n")
            lines.append("\n")
            with open(report_path, 'a', encoding='utf-8') as f:
                f.writelines(lines)

            cum_cm += confusion_matrix(yt, yp, labels=list(range(num_classes)))
            ok += 1

        except Exception as e:
            print(f"  Failed: {e}"); fail.append(fname)

    # summary metrics (computed from accumulated confusion matrix, no pixel iteration)
    if ok > 0:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Cloud Detection Evaluation Summary  ({ok}/{total} images)")
        print(sep)

        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{sep}\n")
            f.write(f"  Cloud Detection Evaluation Summary  ({ok}/{total} images)\n")
            f.write(f"{sep}\n")
            result = metrics_from_cm(cum_cm, file=f)

        print(f"\n  Evaluated {ok} images")
        print(sep)
        print(f"Report saved -> {report_path}")

    if fail:
        print(f"\nFailed files ({len(fail)}):", "\n  ".join(fail))


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
# "cloudnet", "cdnetv2", "hrcloudnet", "mscff", "swinunet", "mcdnet", "rdunet", "cloudmamba"
def main():
    parser = argparse.ArgumentParser(description='WHUS2 binary cloud detection batch inference and evaluation')
    parser.add_argument('--model_name',  type=str, default='cloudmamba',  help='model name (cloudnet / cdnetv2 / hrcloudnet / swinunet / rdunet / cloudmamba)')
    parser.add_argument('--checkpoint',  type=str, default='./checkpoints/cloudmamba/saved_models/2026030818_best_loss_epoch_050.pth', help='path to model checkpoint (.pth)')
    parser.add_argument('--mode',        type=str, default='batch_eval', choices=['single', 'batch', 'batch_eval'],  help='single=one image  batch=batch inference  batch_eval=batch inference+evaluation')
    parser.add_argument('--image_path',  type=str, help='input image path (single mode)')
    parser.add_argument('--output_path', type=str, help='output path (single mode)')
    parser.add_argument('--image_dir',   type=str, default='./data/test/image',  help='image directory (batch / batch_eval mode)')
    parser.add_argument('--label_dir',   type=str, default='./data/test/gt',  help='label directory (batch_eval mode)')
    parser.add_argument('--output_dir',  type=str, default='./results/cloudmamba', help='output directory')
    parser.add_argument('--num_classes', type=int, default=2,  help='number of classes (binary=2)')
    parser.add_argument('--in_channels', type=int, default=4,  help='input channels (R/G/B/NIR=4)')
    parser.add_argument('--img_size',    type=int, default=512, help='model input size')
    parser.add_argument('--patch_size',  type=int, default=512, help='sliding window size (recommended 512)')
    parser.add_argument('--overlap',     type=int, default=64,  help='sliding window overlap in pixels')
    parser.add_argument('--gpu',         type=str, default='0')
    args = parser.parse_args()

    # auto-sync output_dir with model_name if using default
    if args.output_dir == './results/cloudmamba' and args.model_name != 'cloudmamba':
        args.output_dir = f'./results/{args.model_name}'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # build config for get_model
    class Config:
        def __init__(self, a):
            self.model_name = a.model_name
            self.save_name  = a.model_name
            self.img_size   = a.img_size
            self.in_channels = a.in_channels
            self.num_classes = a.num_classes
            self.batch_size  = 1

    config = Config(args)
    model  = get_model(args=config, device=device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    print(f"Checkpoint loaded: {args.checkpoint}")

    t0 = time.time()

    if args.mode == 'single':
        assert args.image_path and args.output_path, \
            "single mode requires --image_path and --output_path"
        predict_single(model, args.image_path, args.output_path, device,
                       args.num_classes, args.in_channels,
                       args.patch_size, args.overlap)

    elif args.mode == 'batch':
        assert args.image_dir, "batch mode requires --image_dir"
        predict_batch(model, args.image_dir, args.output_dir, device,
                      args.num_classes, args.in_channels,
                      args.patch_size, args.overlap)

    elif args.mode == 'batch_eval':
        assert args.image_dir and args.label_dir, \
            "batch_eval mode requires --image_dir and --label_dir"
        predict_batch_with_metrics(model, args.image_dir, args.label_dir,
                                   args.output_dir, device,
                                   args.num_classes, args.in_channels,
                                   args.patch_size, args.overlap)

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
