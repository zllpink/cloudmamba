"""Cloud detection network with multi-scale skip connections and CBAM attention.

Ported from Keras to PyTorch. Architecture:
  - 5-level encoder with residual-shortcut contracting arms
  - Multi-scale feature-fusion skip connections (ImproveFfBlock 0-4)
  - CBAM attention in every decoder block
  - Lightweight refinement sub-network that refines the initial prediction
  - Binary sigmoid output (or multi-class with num_classes > 1)

Input:  (B, in_channels, H, W)  – default 4-channel, H/W must be a multiple of 32
Output: (B, num_classes, H, W)  – sigmoid probability map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Attention ────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ─── Core Building Blocks ─────────────────────────────────────────────────────

class ContrArm(nn.Module):
    """Double-conv + concat-shortcut residual.

    Structural constraint: in_channels == filters // 2
      shortcut = concat(x, conv1x1(x))  →  2 * in_channels = filters
    """

    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
        )
        self.side = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = torch.cat([x, self.side(x)], dim=1)   # 2 * in_ch = filters
        return self.relu(self.main(x) + shortcut)


class Bridge(nn.Module):
    """Same as ContrArm with Dropout before the second BN-ReLU."""

    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.Dropout2d(0.15),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
        )
        self.side = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = torch.cat([x, self.side(x)], dim=1)
        return self.relu(self.main(x) + shortcut)


class ContrArm2(nn.Module):
    """Simple double-conv without residual (used in refinement sub-network)."""

    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpPathBlock(nn.Module):
    """Expanding-path block: Conv → BN-ReLU → CBAM."""

    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, filters, 3, padding=1)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(filters), nn.ReLU(inplace=True))
        self.cbam    = CBAM(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.bn_relu(self.conv(x)))


# ─── Multi-Scale Skip-Connection Blocks ───────────────────────────────────────

def _pool_repeat(x: torch.Tensor, target_size, n: int) -> torch.Tensor:
    """Adaptive-pool to target_size first, then repeat channels (n+1) times.

    Equivalent to F.max_pool2d(_repeat_ch(x, n), stride) but with much lower
    memory overhead (shrink spatial dims before channel repeat).
    """
    return F.adaptive_max_pool2d(x, target_size).repeat(1, n + 1, 1, 1)


class ImproveFfBlock4(nn.Module):
    """Multi-scale skip targeting conv5 spatial resolution (H/32).

    Channel counts per branch (all equal to conv5 channels = 512):
        conv4 (256ch): pool → target, repeat×2 → 512
        conv3 (128ch): pool → target, repeat×4 → 512
        conv2  (64ch): pool → target, repeat×8 → 512
        conv1  (32ch): pool → target, repeat×16→ 512
    """

    def forward(self, conv4, conv3, conv2, conv1, pure_ff):
        sz = pure_ff.shape[2:]
        x1 = _pool_repeat(conv4,  sz,  1)
        x2 = _pool_repeat(conv3,  sz,  3)
        x3 = _pool_repeat(conv2,  sz,  7)
        x4 = _pool_repeat(conv1,  sz, 15)
        return F.relu(x1 + x2 + x3 + x4 + pure_ff)


class ImproveFfBlock3(nn.Module):
    """Multi-scale skip targeting conv4 spatial resolution (H/16).

        conv3 (128ch): pool → target, repeat×2 → 256
        conv2  (64ch): pool → target, repeat×4 → 256
        conv1  (32ch): pool → target, repeat×8 → 256
    """

    def forward(self, conv3, conv2, conv1, pure_ff):
        sz = pure_ff.shape[2:]
        x1 = _pool_repeat(conv3, sz, 1)
        x2 = _pool_repeat(conv2, sz, 3)
        x3 = _pool_repeat(conv1, sz, 7)
        return F.relu(x1 + x2 + x3 + pure_ff)


class ImproveFfBlock2(nn.Module):
    """Multi-scale skip targeting conv3 spatial resolution (H/8).

        conv2  (64ch): pool → target, repeat×2 → 128
        conv1  (32ch): pool → target, repeat×4 → 128
        conv7 (512ch): ConvTranspose(128, stride=4) → 128
    """

    def __init__(self):
        super().__init__()
        self.up_conv7 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=4),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

    def forward(self, conv2, conv1, pure_ff, conv7):
        sz = pure_ff.shape[2:]
        x1 = _pool_repeat(conv2, sz, 1)
        x2 = _pool_repeat(conv1, sz, 3)
        x3 = self.up_conv7(conv7)
        return F.relu(x1 + x2 + pure_ff + x3)


class ImproveFfBlock1(nn.Module):
    """Multi-scale skip targeting conv2 spatial resolution (H/4).

        conv1  (32ch): pool → target, repeat×2 →  64
        conv7 (512ch): ConvTranspose(64, stride=8)  →  64
        conv8 (256ch): ConvTranspose(64, stride=4)  →  64
    """

    def __init__(self):
        super().__init__()
        self.up_conv7 = nn.Sequential(
            nn.ConvTranspose2d(512, 64, kernel_size=8, stride=8),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up_conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

    def forward(self, conv1, pure_ff, conv7, conv8):
        sz = pure_ff.shape[2:]
        x1 = _pool_repeat(conv1, sz, 1)
        x2 = self.up_conv7(conv7)
        x3 = self.up_conv8(conv8)
        return F.relu(x1 + pure_ff + x2 + x3)


class ImproveFfBlock0(nn.Module):
    """Multi-scale skip targeting conv1 spatial resolution (H/2).

        conv7 (512ch): ConvTranspose(32, stride=16) →  32
        conv8 (256ch): ConvTranspose(32, stride=8)  →  32
        conv9 (128ch): ConvTranspose(32, stride=4)  →  32
    """

    def __init__(self):
        super().__init__()
        self.up_conv7 = nn.Sequential(
            nn.ConvTranspose2d(512, 32, kernel_size=16, stride=16),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.up_conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 32, kernel_size=8, stride=8),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.up_conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

    def forward(self, pure_ff, conv7, conv8, conv9):
        x1 = self.up_conv7(conv7)
        x2 = self.up_conv8(conv8)
        x3 = self.up_conv9(conv9)
        return F.relu(pure_ff + x1 + x2 + x3)


# ─── Main Model ───────────────────────────────────────────────────────────────

class CloudDetNet(nn.Module):
    """Cloud detection network with dual-decoder and refinement sub-network.

    Encoder:    5-level ContrArm + Bridge bottleneck
    Decoder:    5 upsampling stages, each with ImproveFfBlock skip + CBAM ExpPathBlock
    Refinement: small 3-level U-Net that refines the initial prediction
    Output:     sigmoid probability map

    Args:
        in_channels: spectral channels of input image (default 4)
        num_classes:  1 → binary sigmoid; >1 → multi-class sigmoid per channel
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, out_channels: int = None):
        super().__init__()
        if out_channels is not None:
            num_classes = out_channels

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stem  = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.enc1  = ContrArm(16, 32)       # 32ch, H/2  (after pool)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = ContrArm(32, 64)       # 64ch, H/4
        self.pool2 = nn.MaxPool2d(2)
        self.enc3  = ContrArm(64, 128)      # 128ch, H/8
        self.pool3 = nn.MaxPool2d(2)
        self.enc4  = ContrArm(128, 256)     # 256ch, H/16
        self.pool4 = nn.MaxPool2d(2)
        self.enc5  = ContrArm(256, 512)     # 512ch, H/32
        self.pool5 = nn.MaxPool2d(2)
        self.bridge = Bridge(512, 1024)     # 1024ch, H/32 (bottleneck)

        # ── Primary Decoder ───────────────────────────────────────────────────
        self.up7   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.ff4   = ImproveFfBlock4()
        self.dec7  = ExpPathBlock(1024, 512)    # cat(up7, ff4) → 1024 in

        self.up8   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ff3   = ImproveFfBlock3()
        self.dec8  = ExpPathBlock(512, 256)

        self.up9   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ff2   = ImproveFfBlock2()
        self.dec9  = ExpPathBlock(256, 128)

        self.up10  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ff1   = ImproveFfBlock1()
        self.dec10 = ExpPathBlock(128, 64)

        self.up11  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ff0   = ImproveFfBlock0()
        self.dec11 = ExpPathBlock(64, 32)

        self.head1 = nn.Conv2d(32, num_classes, 1)

        # ── Refinement Sub-Network ────────────────────────────────────────────
        self.ref_enc1   = ContrArm2(num_classes + 32, 32)
        self.ref_pool1  = nn.MaxPool2d(2)
        self.ref_enc2   = ContrArm2(32, 64)
        self.ref_pool2  = nn.MaxPool2d(2)
        self.ref_enc3   = ContrArm2(64, 128)
        self.ref_pool3  = nn.MaxPool2d(2)
        self.ref_bridge = Bridge(128, 256)

        self.ref_up1  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ref_dec1 = ExpPathBlock(256, 128)

        self.ref_up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ref_dec2 = ExpPathBlock(128, 64)

        self.ref_up3  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ref_dec3 = ExpPathBlock(64, 32)

        self.head2      = nn.Conv2d(32, num_classes, 1)
        self.head_final = nn.Conv2d(num_classes, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──────────────────────────────────────────────────────────
        s  = self.stem(x)
        c1 = self.enc1(s)                       # 32ch,  H/2
        c2 = self.enc2(self.pool1(c1))          # 64ch,  H/4
        c3 = self.enc3(self.pool2(c2))          # 128ch, H/8
        c4 = self.enc4(self.pool3(c3))          # 256ch, H/16
        c5 = self.enc5(self.pool4(c4))          # 512ch, H/32
        c6 = self.bridge(self.pool5(c5))        # 1024ch, H/64

        # ── Decoder stage 7 ──────────────────────────────────────────────────
        t7 = self.up7(c6)                       # 512ch, H/32
        p7 = self.ff4(c4, c3, c2, c1, c5)      # 512ch, H/32
        c7 = self.dec7(torch.cat([t7, p7], dim=1))
        c7 = F.relu(c7 + c5 + t7)              # add_block_exp_path

        # ── Decoder stage 8 ──────────────────────────────────────────────────
        t8 = self.up8(c7)                       # 256ch, H/16
        p8 = self.ff3(c3, c2, c1, c4)          # 256ch, H/16
        c8 = self.dec8(torch.cat([t8, p8], dim=1))
        c8 = F.relu(c8 + c4 + t8)

        # ── Decoder stage 9 ──────────────────────────────────────────────────
        t9 = self.up9(c8)                       # 128ch, H/8
        p9 = self.ff2(c2, c1, c3, c7)          # 128ch, H/8
        c9 = self.dec9(torch.cat([t9, p9], dim=1))
        c9 = F.relu(c9 + c3 + t9)

        # ── Decoder stage 10 ─────────────────────────────────────────────────
        t10 = self.up10(c9)                     # 64ch, H/4
        p10 = self.ff1(c1, c2, c7, c8)         # 64ch, H/4
        c10 = self.dec10(torch.cat([t10, p10], dim=1))
        c10 = F.relu(c10 + c2 + t10)

        # ── Decoder stage 11 ─────────────────────────────────────────────────
        t11 = self.up11(c10)                    # 32ch, H/2
        p11 = self.ff0(c1, c7, c8, c9)         # 32ch, H/2
        c11 = self.dec11(torch.cat([t11, p11], dim=1))
        c11 = F.relu(c11 + c1 + t11)

        pred1 = self.head1(c11)                 # num_classes ch, H/2

        # ── Refinement Sub-Network ────────────────────────────────────────────
        r     = torch.cat([pred1, c1], dim=1)  # (num_classes+32)ch
        rc1   = self.ref_enc1(r)               # 32ch
        rc2   = self.ref_enc2(self.ref_pool1(rc1))   # 64ch
        rc3   = self.ref_enc3(self.ref_pool2(rc2))   # 128ch
        rc4   = self.ref_bridge(self.ref_pool3(rc3)) # 256ch

        ru1   = self.ref_up1(rc4)
        rc5   = self.ref_dec1(torch.cat([ru1, rc3], dim=1))   # 128ch

        ru2   = self.ref_up2(rc5)
        rc6   = self.ref_dec2(torch.cat([ru2, rc2], dim=1))   # 64ch

        ru3   = self.ref_up3(rc6)
        rc7   = self.ref_dec3(torch.cat([ru3, rc1], dim=1))   # 32ch

        pred2 = self.head2(rc7)                # num_classes ch

        # ── Final output (logits, compatible with CrossEntropyLoss) ──────────
        return self.head_final(pred1 + pred2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CloudDetNet(in_channels=3, num_classes=2).to(device)
    x      = torch.randn(2, 3, 512, 512).to(device)
    y      = model(x)
    print(f"Output shape : {y.shape}")   # expected: torch.Size([2, 2, 512, 512])
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params / 1e6:.2f} M")
