import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ─── Sobel Kernel (for boundary / thin-cloud) ─────────────────────────────────
def _sobel_edges(mask: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradient magnitude of a float mask [B,1,H,W]."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(mask, kx, padding=1)
    gy = F.conv2d(mask, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


class BoundaryLoss(nn.Module):
    """Boundary F1 loss (Bokhovkin et al., arXiv 1905.07852).

    Fixes vs. original:
      - Uses unsqueeze(0) guard so batch-size > 1 works correctly.
      - One-hot built on the same device as pred (no explicit .cuda() calls).
      - theta0 / theta are forced odd so max_pool padding is exact.

    Args:
        theta0: inner dilation radius for boundary extraction (pixels).
        theta:  outer dilation radius for boundary matching tolerance.
    """

    def __init__(self, theta0: int = 3, theta: int = 5):
        super().__init__()
        # ensure kernels are odd
        self.theta0 = theta0 + (1 - theta0 % 2)
        self.theta  = theta  + (1 - theta  % 2)

    def _one_hot(self, targets: torch.Tensor, n_cls: int) -> torch.Tensor:
        """targets: (N,H,W) long → (N,n_cls,H,W) float on same device."""
        N, H, W = targets.shape
        oh = torch.zeros(N, n_cls, H, W, dtype=torch.float32, device=targets.device)
        oh.scatter_(1, targets.unsqueeze(1).long(), 1.0)
        return oh

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred : (N, C, H, W) logits
        gt   : (N, H, W)   integer labels
        """
        N, C, H, W = pred.shape
        prob = F.softmax(pred, dim=1)                    # (N,C,H,W)
        one_hot = self._one_hot(gt, C)                   # (N,C,H,W)

        p0, p1 = (self.theta0 - 1) // 2, (self.theta - 1) // 2

        # boundary maps: where foreground transitions to background
        gt_b   = F.max_pool2d(1 - one_hot, self.theta0, stride=1, padding=p0) - (1 - one_hot)
        pred_b = F.max_pool2d(1 - prob,    self.theta0, stride=1, padding=p0) - (1 - prob)

        # extended boundary maps for tolerance matching
        gt_b_ext   = F.max_pool2d(gt_b,   self.theta, stride=1, padding=p1)
        pred_b_ext = F.max_pool2d(pred_b, self.theta, stride=1, padding=p1)

        gt_b   = gt_b.view(N, C, -1)
        pred_b = pred_b.view(N, C, -1)
        gt_b_ext   = gt_b_ext.view(N, C, -1)
        pred_b_ext = pred_b_ext.view(N, C, -1)

        P   = (pred_b * gt_b_ext).sum(2) / (pred_b.sum(2) + 1e-7)
        R   = (pred_b_ext * gt_b).sum(2) / (gt_b.sum(2)   + 1e-7)
        BF1 = 2 * P * R / (P + R + 1e-7)

        return torch.mean(1.0 - BF1)


class ThinCloudLoss(nn.Module):
    """Thin-cloud detection auxiliary loss.

    Thin clouds are semi-transparent and low-contrast – they are the hardest
    class for standard CE/Dice losses because their gradient signal is small.
    This loss amplifies supervision in two complementary ways:

    1. **Edge-weighted CE**: pixels near cloud boundaries (detected via Sobel
       on the GT mask) receive a higher loss weight so the model learns crisp
       thin-cloud contours.

    2. **Low-density region emphasis**: inside the cloud mask, pixels where the
       *image* is bright (high mean reflectance) are down-weighted (likely thick
       cloud, already easy), while dim cloud pixels are up-weighted (likely thin
       cloud, hard).  This is implemented as an inverse-brightness weight map
       applied only to cloud-labelled pixels.

    Args:
        edge_weight   : extra weight multiplier on boundary pixels (≥1).
        thin_weight   : weight multiplier on dim-cloud (thin-cloud) pixels.
        edge_radius   : dilation radius (pixels) around GT boundary.
        brightness_thr: reflectance percentile below which a cloud pixel is
                        considered "thin" (0–1, applied per sample).
    """

    def __init__(self,
                 edge_weight:    float = 3.0,
                 thin_weight:    float = 2.0,
                 edge_radius:    int   = 3,
                 brightness_thr: float = 0.4):
        super().__init__()
        self.edge_weight    = edge_weight
        self.thin_weight    = thin_weight
        self.edge_radius    = edge_radius + (1 - edge_radius % 2)   # force odd
        self.brightness_thr = brightness_thr

    def forward(self,
                pred:  torch.Tensor,   # (N, C, H, W) logits
                gt:    torch.Tensor,   # (N, H, W) int labels
                image: torch.Tensor,   # (N, 3, H, W) normalised RGB in [-1,1]
                ) -> torch.Tensor:

        N, C, H, W = pred.shape
        gt_f = gt.float().unsqueeze(1)                      # (N,1,H,W)

        # ── 1. boundary weight map ──────────────────────────────────────────
        edge = _sobel_edges(gt_f)                           # (N,1,H,W)
        # dilate
        pad  = (self.edge_radius - 1) // 2
        edge = F.max_pool2d(edge, self.edge_radius, stride=1, padding=pad)
        # normalise to [0, 1] per sample
        edge_max = edge.flatten(1).max(dim=1).values.view(N, 1, 1, 1).clamp(min=1e-6)
        edge_norm = (edge / edge_max).squeeze(1)            # (N, H, W)
        # weight: 1 everywhere, up to edge_weight at boundaries
        w_edge = 1.0 + (self.edge_weight - 1.0) * edge_norm  # (N, H, W)

        # ── 2. thin-cloud weight map (dim-cloud emphasis) ───────────────────
        # brightness: mean over RGB channels, mapped from [-1,1] → [0,1]
        brightness = ((image + 1.0) / 2.0).mean(dim=1, keepdim=True)  # (N,1,H,W)
        cloud_mask = gt_f                                   # 1 where cloud

        # per-sample brightness threshold for "thin cloud"
        # use brightness_thr as a fixed threshold on [0,1] reflectance
        is_thin = (brightness < self.brightness_thr) & (cloud_mask > 0.5)
        w_thin  = torch.where(is_thin,
                              torch.full_like(brightness, self.thin_weight),
                              torch.ones_like(brightness)).squeeze(1)  # (N,H,W)

        # ── 3. combined pixel-wise weight ───────────────────────────────────
        pixel_weight = w_edge * w_thin                      # (N, H, W)

        # ── 4. weighted cross-entropy ────────────────────────────────────────
        log_p = F.log_softmax(pred, dim=1)                  # (N, C, H, W)
        # gather log-prob for the ground-truth class at each pixel
        log_p_gt = log_p.gather(1, gt.long().unsqueeze(1)).squeeze(1)  # (N,H,W)
        loss = -(pixel_weight * log_p_gt).mean()

        return loss


class DiceLoss(nn.Module):
    """Dice Loss for binary or multiclass segmentation.

    Works on softmax logits (N, C, H, W) + integer labels (N, H, W).
    For binary cloud detection (C=2) this penalises FP/FN symmetrically,
    helping with cloud/clear-sky class imbalance.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)                          # (N, C, H, W)
        # one-hot encode targets
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1).long(), 1.0)    # (N, C, H, W)

        probs_flat = probs.view(n, c, -1)
        oh_flat    = one_hot.view(n, c, -1)

        intersection = (probs_flat * oh_flat).sum(dim=2)          # (N, C)
        cardinality  = probs_flat.sum(dim=2) + oh_flat.sum(dim=2) # (N, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class SimpleCloudCELoss(nn.Module):
    """Simple Cross-Entropy loss for cloud detection.

    Expected inputs:
        logits : (N, C, H, W), usually C=2 for binary cloud/clear segmentation
        targets: (N, H, W), integer labels in {0, 1}
    """

    def __init__(self, class_weight: torch.Tensor = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets.long())


class CEDiceLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss for binary cloud detection.

    weight_ce  : weight for CrossEntropyLoss term
    weight_dice: weight for DiceLoss term
    class_weight: optional tensor of shape (num_classes,) for class imbalance
    """

    def __init__(self, weight_ce: float = 1.0, weight_dice: float = 1.0,
                 class_weight: torch.Tensor = None):
        super().__init__()
        self.weight_ce   = weight_ce
        self.weight_dice = weight_dice
        self.ce   = nn.CrossEntropyLoss(weight=class_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce   = self.ce(logits, targets.long())
        loss_dice = self.dice(logits, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice


#miou loss
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

#Minimax iou
class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        #minimum iou of two classes
        min_iou = torch.min(iou)

        #loss
        loss = -min_iou-torch.mean(iou)
        return loss

