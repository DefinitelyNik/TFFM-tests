"""Explicit per-image metrics at 512x512. Topology tolerances are reconstruction
choices, not recovered author settings. No FOV masking or postprocessing.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


def components(x):
    return ndi.label(x, structure=np.ones((3,3), dtype=bool))[1]


def safe_overlap(p, g):
    tp = int(np.logical_and(p,g).sum())
    npred, ngt = int(p.sum()), int(g.sum())
    if npred + ngt == 0:
        return 1., 1., 1., 1.
    dice = 2*tp / (npred+ngt)
    union = npred+ngt-tp
    iou = tp / union if union else 1.
    precision = tp/npred if npred else 0.
    recall = tp/ngt if ngt else 0.
    return dice, iou, precision, recall


def hd95(p, g):
    if not p.any() and not g.any():
        return 0.
    if not p.any() or not g.any():
        return float('inf')
    conn = ndi.generate_binary_structure(2, 1)
    ps = p ^ ndi.binary_erosion(p, structure=conn, border_value=0)
    gs = g ^ ndi.binary_erosion(g, structure=conn, border_value=0)
    dp = ndi.distance_transform_edt(~gs)[ps]
    dg = ndi.distance_transform_edt(~ps)[gs]
    # Max of directed percentiles, not percentile of concatenated distances.
    return float(max(np.percentile(dp,95), np.percentile(dg,95)))


def topology_cldice(p, g, sp, sg):
    if not p.any() and not g.any():
        return 1.
    if not sp.any() or not sg.any():
        return 0.
    prec = float((sp & g).sum()) / sp.sum()
    sens = float((sg & p).sum()) / sg.sum()
    return float(2*prec*sens/(prec+sens)) if prec+sens else 0.


def skeleton_f1(sp, sg, tolerance=2.):
    if not sp.any() and not sg.any():
        return 1.
    if not sp.any() or not sg.any():
        return 0.
    precision = float((ndi.distance_transform_edt(~sg)[sp] <= tolerance).mean())
    recall = float((ndi.distance_transform_edt(~sp)[sg] <= tolerance).mean())
    return 2*precision*recall/(precision+recall) if precision+recall else 0.


def junctions(s):
    neighbors = ndi.convolve(s.astype(np.int32), np.ones((3,3), dtype=np.int32),
                             mode='constant', cval=0) - s
    mask = s & (neighbors >= 3)
    labels, n = ndi.label(mask, structure=np.ones((3,3)))
    if not n:
        return np.empty((0,2))
    return np.asarray(ndi.center_of_mass(mask, labels, range(1,n+1)))


def junction_scores(sp, sg, tolerance=3.):
    p, g = junctions(sp), junctions(sg)
    if len(p) == 0 and len(g) == 0:
        return 1., 1., 1.
    matches = 0
    if len(p) and len(g):
        dist = cdist(p,g)
        # High cost for forbidden edges maximizes feasible match cardinality.
        cost = np.where(dist <= tolerance, dist, (len(p)+len(g)+1)*(tolerance+1))
        a,b = linear_sum_assignment(cost)
        matches = int((dist[a,b] <= tolerance).sum())
    prec = matches/len(p) if len(p) else 0.
    rec = matches/len(g) if len(g) else 0.
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.
    return prec, rec, f1


def binary_metrics(p, g, skeleton_tolerance=2., junction_tolerance=3.):
    p, g = np.asarray(p, bool), np.asarray(g, bool)
    dice, iou, prec, rec = safe_overlap(p,g)
    sp, sg = skeletonize(p), skeletonize(g)
    jp,jr,jf = junction_scores(sp,sg,junction_tolerance)
    cp,cg = components(p), components(g)
    return dict(dice=float(dice), iou=float(iou), precision=float(prec), recall=float(rec),
                accuracy=float(np.mean(p == g)),
                hd95=hd95(p,g), cldice=topology_cldice(p,g,sp,sg),
                betti0_error=abs(cp-cg), pred_components=cp, gt_components=cg,
                skeleton_f1=skeleton_f1(sp,sg,skeleton_tolerance),
                junction_precision=jp,junction_recall=jr,junction_f1=jf)
