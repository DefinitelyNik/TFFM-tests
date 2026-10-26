"""Manifest-based data loading: explicit A/V masks, no guessed label colours."""
import csv
import random
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from monai.transforms import (Compose, RandRotated, RandFlipd, RandAffined,
    RandAdjustContrastd, RandShiftIntensityd, RandGaussianNoised, RandGaussianSmoothd)


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'id', 'image', 'artery', 'vein', 'split'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('CSV requires columns: id,image,artery,vein,split; optional group')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample ids')
    crossing_flags = [bool(r.get('crossing', '').strip()) for r in rows]
    if any(crossing_flags) and not all(crossing_flags):
        raise ValueError('Provide crossing mask for every image or for none')
    path_keys = ('image', 'artery', 'vein') + (('crossing',) if all(crossing_flags) else ())
    seen_paths, seen_groups = {}, {}
    for r in rows:
        if r['split'] not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid split for {r['id']}")
        for key in path_keys:
            p = Path(r[key])
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            r[key] = str(p)
        if r['image'] in seen_paths:
            raise ValueError('Repeated image path; potential leakage')
        seen_paths[r['image']] = r['split']
        group = r.get('group', '').strip()
        if group:
            if group in seen_groups and seen_groups[group] != r['split']:
                raise ValueError(f'Patient/group leakage: {group}')
            seen_groups[group] = r['split']
    return rows


def write_manifest(rows, output):
    fields = ['id', 'image', 'artery', 'vein', 'crossing', 'split', 'group']
    output = Path(output).resolve()
    with output.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in fields}
            for key in ('image', 'artery', 'vein', 'crossing'):
                if not row.get(key, '').strip():
                    continue
                p = Path(row[key])
                # Internal callers provide resolved paths. Keep paths relative to
                # this CSV so moving the complete project preserves references.
                if p.is_absolute():
                    row[key] = Path(os.path.relpath(p, output.parent)).as_posix()
            w.writerow(row)


def generate_split(source, output, seed):
    # Input uses the same columns; split values are ignored and can be empty.
    source = Path(source).resolve()
    with source.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'id', 'image', 'artery', 'vein'}.issubset(rows[0]):
        raise ValueError('Input needs id,image,artery,vein columns')
    crossing_flags = [bool(r.get('crossing', '').strip()) for r in rows]
    if any(crossing_flags) and not all(crossing_flags):
        raise ValueError('Provide crossing mask for every image or for none')
    has_group = [bool(r.get('group', '').strip()) for r in rows]
    if any(has_group) and not all(has_group):
        raise ValueError('Provide group for every image or for none')
    for r in rows:
        for k in ('image', 'artery', 'vein') + (('crossing',) if all(crossing_flags) else ()):
            p = Path(r[k])
            r[k] = str((source.parent / p).resolve() if not p.is_absolute() else p.resolve())
    units = sorted({r['group'] if all(has_group) else r['id'] for r in rows})
    random.Random(seed).shuffle(units)
    nval = max(1, round(len(units) * .1))
    ntest = max(1, round(len(units) * .1))
    if len(units) - nval - ntest < 1:
        raise ValueError('Need at least 3 independent images/groups')
    assignments = {u: ('test' if i < ntest else 'val' if i < ntest+nval else 'train')
                   for i, u in enumerate(units)}
    for r in rows:
        r['split'] = assignments[r['group'] if all(has_group) else r['id']]
    write_manifest(rows, output)
    read_manifest(output)
    return {s: sum(r['split'] == s for r in rows) for s in ('train','val','test')}


def augmentation(seed):
    t = Compose([
        RandRotated(keys=['image','mask','crossing'], range_x=np.pi/6, prob=.5,
                    keep_size=True, mode=('bilinear','nearest','nearest'), padding_mode='zeros'),
        RandFlipd(keys=['image','mask','crossing'], spatial_axis=0, prob=.5),
        RandFlipd(keys=['image','mask','crossing'], spatial_axis=1, prob=.5),
        RandAffined(keys=['image','mask','crossing'], prob=.3, spatial_size=(512,512),
                    translate_range=(25.6,25.6), mode=('bilinear','nearest','nearest'),
                    padding_mode='zeros'),
        RandAdjustContrastd(keys='image', prob=.3, gamma=(.8,1.2)),
        RandShiftIntensityd(keys='image', prob=.3, offsets=.1),
        RandGaussianNoised(keys='image', prob=.2, mean=0, std=.01, sample_std=False),
        RandGaussianSmoothd(keys='image', prob=.2, sigma_x=(.5,1.), sigma_y=(.5,1.)),
    ])
    t.set_random_state(seed=seed)
    return t


def binary_mask(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2 or not np.isin(np.unique(a), [0,1,255]).all():
        raise ValueError(f'{path}: expected single-channel binary mask with values 0/1 or 0/255')
    return a > 0


class AVDataset(Dataset):
    def __init__(self, rows, training=False, seed=42, intensity='minmax'):
        self.rows = rows
        self.transforms = augmentation(seed) if training else None
        self.intensity = intensity

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(r['image']) as im:
            im = im.convert('RGB')
            original_size = (im.height, im.width)
            x = np.asarray(im.resize((512,512), Image.Resampling.BILINEAR),
                           dtype=np.float32).copy()
        masks = []
        for key in ('artery', 'vein'):
            a = binary_mask(r[key])
            if a.shape != original_size:
                raise ValueError(f"Original image/mask size mismatch: {r['id']}, {key}")
            a = Image.fromarray(a.astype(np.uint8)).resize((512,512), Image.Resampling.NEAREST)
            masks.append(np.asarray(a, dtype=np.float32).copy())
        if r.get('crossing', '').strip():
            crossing = binary_mask(r['crossing'])
            if crossing.shape != original_size:
                raise ValueError(f"Original image/mask size mismatch: {r['id']}, crossing")
            crossing = np.asarray(Image.fromarray(crossing.astype(np.uint8)).resize(
                (512,512), Image.Resampling.NEAREST), dtype=np.float32).copy()
        else:
            # Backward-compatible fallback for old manifests produced by the
            # two-channel converter: green crossings were written to BOTH masks.
            crossing = masks[0] * masks[1]
        if self.intensity == 'minmax':
            lo, hi = x.min(), x.max()
            x = (x - lo) / max(float(hi-lo), 1e-8)
        elif self.intensity == 'uint8':
            x = x / 255.
        else:
            raise ValueError(self.intensity)
        d = {'image': torch.from_numpy(x.transpose(2,0,1).copy()),
             'mask': torch.from_numpy(np.stack(masks)),
             'crossing': torch.from_numpy(crossing[None].copy())}
        if self.transforms:
            d = self.transforms(d)
        # No clipping after noise/brightness: explicit pipeline choice.
        return {'image': torch.as_tensor(d['image']).float(),
                'mask': torch.as_tensor(d['mask']).float(),
                'crossing': torch.as_tensor(d['crossing']).float(), 'id': r['id']}
