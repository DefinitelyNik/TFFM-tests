"""Supplied repository loss: computational behaviour intentionally preserved.
This is NOT the mathematical clDice/Tversky objective printed in the paper.
Use paper_losses.PaperLoss for an explicitly equation-based alternative.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import TverskyLoss


# ============= SOFT SKELETONIZATION =============
class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=10, beta=15.0):
        super().__init__()
        self.num_iter, self.beta = num_iter, beta

    def forward(self, img):
        if img.min() < 0 or img.max() > 1.0:
            img = torch.sigmoid(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        B, C, H, W = img.shape
        skeleton, eroded = torch.zeros_like(img), img.clone()
        kernel = torch.ones((C, 1, 3, 3), device=img.device, dtype=img.dtype) / 9.0

        for i in range(self.num_iter):
            eroded_prev = eroded
            eroded_smooth = F.conv2d(eroded, kernel, padding=1, groups=C)
            weights = F.softmin(
                torch.stack([eroded, eroded_smooth], dim=0) * self.beta, dim=0
            )
            eroded = weights[0] * eroded + weights[1] * eroded_smooth
            skeleton += F.relu(eroded_prev - eroded)
            if torch.abs(eroded - eroded_prev).mean() < 1e-6:
                break

        return skeleton / (self.num_iter + 1)


# ============= CLDICE LOSS =============
class CLDiceLoss(nn.Module):
    def __init__(self, num_iter=10, weight=0.4):
        super().__init__()
        self.skeletonize, self.weight = SoftSkeletonize(num_iter=num_iter), weight

    def forward(self, pred, target):
        skel_pred = self.skeletonize(pred)
        skel_target = self.skeletonize(target.float())

        intersect_prec = (skel_pred * target).sum(dim=[2, 3]) + 1e-7
        union_prec = skel_pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + 1e-7
        precision = (2.0 * intersect_prec) / (union_prec + intersect_prec)

        pred_prob = (
            pred if pred.min() >= 0 and pred.max() <= 1.0 else torch.sigmoid(pred)
        )
        intersect_rec = (skel_target * pred_prob).sum(dim=[2, 3]) + 1e-7
        union_rec = skel_target.sum(dim=[2, 3]) + pred_prob.sum(dim=[2, 3]) + 1e-7
        recall = (2.0 * intersect_rec) / (union_rec + intersect_rec)

        clDice = 2.0 * precision * recall / (precision + recall + 1e-7)
        return self.weight * (1.0 - clDice.mean())


# ============= COMBINED LOSS =============
class CombinedTverskyCLDice(nn.Module):
    """
    Combines MONAI's TverskyLoss with topology-preserving clDice loss
    Args:
        tversky_alpha: Tversky alpha parameter
        tversky_beta:  Tversky beta parameter
        cldice_weight: Weight for clDice loss
        cldice_iter:   Skeletonization iterations
    """

    def __init__(self, tversky_alpha=0.3, cldice_weight=0.5, cldice_iter=5):
        super().__init__()
        self.tversky_loss = TverskyLoss(
            sigmoid=True,  # For multilabel [B, 2, H, W]
            alpha=tversky_alpha,
        )
        self.cldice_loss = CLDiceLoss(num_iter=cldice_iter, weight=cldice_weight)

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, 2, H, W] - model logits
            target: [B, 2, H, W] - ground truth (0/1)
        Returns:
            Scalar loss value
        """
        # Clamp target to valid range
        target = target.clamp(0.0, 1.0)

        # Compute both losses
        loss_tversky = self.tversky_loss(pred, target)
        loss_cldice = self.cldice_loss(pred, target)

        return loss_tversky + loss_cldice


loss_fn = CombinedTverskyCLDice(tversky_alpha=0.65, cldice_weight=0.5, cldice_iter=5)
