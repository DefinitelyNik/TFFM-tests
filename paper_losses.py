"""Loss matching equations 8-13 of the supplied TFFM paper, not loss.py.
MONAI alpha weights FP and beta weights FN (opposite names to paper Eq.8).
"""
import torch
from torch import nn
from torch.nn import functional as F
from monai.losses import TverskyLoss


def soft_erode(x):
    a = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    b = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.minimum(a, b)


def soft_open(x):
    return F.max_pool2d(soft_erode(x), 3, 1, 1)


def soft_skeleton(x, iterations=5):
    skel = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class PaperLoss(nn.Module):
    def __init__(self, iterations=5, topology_weight=0.5):
        super().__init__()
        # Eq.8: 0.65*FN + 0.35*FP; MONAI does NOT set beta=1-alpha.
        self.region = TverskyLoss(sigmoid=True, alpha=0.35, beta=0.65,
                                  smooth_nr=1e-5, smooth_dr=1e-5,
                                  batch=False, reduction='mean')
        self.iterations = iterations
        self.topology_weight = topology_weight

    def forward(self, logits, target):
        # Explicit sigmoid even when logits happen to lie entirely in [0,1].
        p = logits.float().sigmoid()
        g = target.float()
        sp = soft_skeleton(p, self.iterations)
        sg = soft_skeleton(g, self.iterations)
        dims = (2, 3)
        eps = 1e-7
        tp = ((sp * g).sum(dims) + eps) / (sp.sum(dims) + eps)
        ts = ((sg * p).sum(dims) + eps) / (sg.sum(dims) + eps)
        cl = (2 * tp * ts + eps) / (tp + ts + eps)
        return self.region(logits.float(), g) + self.topology_weight * (1 - cl.mean())


def make_loss(mode):
    if mode == 'paper':
        return PaperLoss()
    if mode == 'repository':
        # Keep supplied file intact: alpha=.65 (FP), beta=.5 MONAI default,
        # and the supplied nonstandard skeleton/clDice implementation.
        from loss import CombinedTverskyCLDice
        return CombinedTverskyCLDice(tversky_alpha=0.65,
                                     cldice_weight=0.5, cldice_iter=5)
    raise ValueError(f'Unknown loss mode: {mode}')
