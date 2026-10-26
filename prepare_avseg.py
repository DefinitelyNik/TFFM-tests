"""Convert Fundus-AVSeg RGB annotations into two binary masks.

Expected layout: ROOT/raw/images and ROOT/raw/annotation.
Crossings (green) are assigned to BOTH channels by this converter.
White/uncertain policy MUST be specified explicitly. These policies are
reconstruction choices, not verified preprocessing settings of TFFM.

Examples:
    python prepare_avseg.py --root datasets/fundus_avseg --white-policy background
    python prepare_avseg.py --root datasets/fundus_avseg --white-policy both

The background policy does NOT ignore pixels in the loss: white pixels become
negative targets in both channels. The both policy labels them positive in
both channels, without resolving their true artery/vein identity.
Outputs are never overwritten. Validation of all input pairs precedes writing.

Use `--export-crossing` to additionally write an independent crossing mask.
Without it, the legacy two-channel export remains unchanged and the evaluator
can only use the explicit `artery AND vein` fallback for crossings.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

COLORS = {
    (0, 0, 0): 'background',
    (255, 0, 0): 'artery',
    (0, 0, 255): 'vein',
    (0, 255, 0): 'crossing',
    (255, 255, 255): 'uncertain',
}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def index_images(folder):
    result = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        key = p.stem.casefold()
        if key in result:
            raise ValueError(f'Duplicate image stem: {result[key]} and {p}')
        result[key] = p
    return result


def read_annotation(path):
    with Image.open(path) as im:
        if im.mode not in ('RGB', 'P', 'RGBA'):
            raise ValueError(f'{path}: expected RGB/palette annotation, got {im.mode}')
        if im.mode == 'RGBA' and np.any(np.asarray(im.getchannel('A')) != 255):
            raise ValueError(f'{path}: non-opaque alpha channel')
        return np.asarray(im.convert('RGB')).copy()


def color_counts(a):
    # Pack RGB to avoid sorting a three-column array for every full-size mask.
    x = a.astype(np.uint32)
    packed = (x[..., 0] << 16) | (x[..., 1] << 8) | x[..., 2]
    values, counts = np.unique(packed, return_counts=True)
    result = {name: 0 for name in COLORS.values()}
    for value, count in zip(values.tolist(), counts.tolist()):
        rgb = ((value >> 16) & 255, (value >> 8) & 255, value & 255)
        if rgb not in COLORS:
            raise ValueError(f'Unexpected RGB color {rgb} ({count} pixels)')
        result[COLORS[rgb]] = int(count)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/fundus_avseg'))
    parser.add_argument('--white-policy', choices=('background', 'both', 'error'), required=True)
    parser.add_argument('--export-crossing', action='store_true',
                        help='Write independent crossing masks and crossing column')
    parser.add_argument('--expected-count', type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    images_dir, annotation_dir = root / 'raw/images', root / 'raw/annotation'
    if not images_dir.is_dir() or not annotation_dir.is_dir():
        raise ValueError(f'Expected folders: {images_dir} and {annotation_dir}')
    images = index_images(images_dir)
    annotations = index_images(annotation_dir)
    if len(annotations) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} annotations, found {len(annotations)}')
    if set(images) != set(annotations):
        raise ValueError(
            f'Images without annotation: {sorted(set(images)-set(annotations))}; '
            f'annotations without image: {sorted(set(annotations)-set(images))}')
    outputs = [root / 'arteries', root / 'veins', root / 'samples.csv', root / 'conversion_report.json']
    if args.export_crossing:
        outputs.append(root / 'crossings')
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite existing outputs: ' + ', '.join(existing))

    totals = {name: 0 for name in COLORS.values()}
    report = []
    print('Validating ALL images, annotation colors and dimensions...')
    for key in sorted(annotations):
        path = annotations[key]
        a = read_annotation(path)
        with Image.open(images[key]) as im:
            if im.size != (a.shape[1], a.shape[0]):
                raise ValueError(f'Dimension mismatch: {images[key]} and {path}')
        try:
            counts = color_counts(a)
        except ValueError as exc:
            raise ValueError(f'{path}: {exc}') from exc
        if args.white_policy == 'error' and counts['uncertain']:
            raise ValueError(f'{path}: uncertain pixels found; explicitly select background or both')
        for name, count in counts.items():
            totals[name] += count
        report.append({'id': path.stem, 'image': str(images[key]),
                       'annotation': str(path), 'pixel_counts': counts})

    (root / 'arteries').mkdir()
    (root / 'veins').mkdir()
    if args.export_crossing:
        (root / 'crossings').mkdir()
    rows = []
    for key in sorted(annotations):
        path = annotations[key]
        a = read_annotation(path)
        red = np.all(a == (255, 0, 0), axis=-1)
        blue = np.all(a == (0, 0, 255), axis=-1)
        green = np.all(a == (0, 255, 0), axis=-1)
        white = np.all(a == (255, 255, 255), axis=-1)
        artery, vein = red | green, blue | green
        if args.white_policy == 'both':
            artery |= white
            vein |= white
        artery_path = root / 'arteries' / (path.stem + '.png')
        vein_path = root / 'veins' / (path.stem + '.png')
        Image.fromarray(artery.astype(np.uint8) * 255).save(artery_path)
        Image.fromarray(vein.astype(np.uint8) * 255).save(vein_path)
        row = {'id': path.stem, 'image': relative(images[key]),
               'artery': relative(artery_path), 'vein': relative(vein_path)}
        if args.export_crossing:
            crossing_path = root / 'crossings' / (path.stem + '.png')
            Image.fromarray(green.astype(np.uint8) * 255).save(crossing_path)
            row['crossing'] = relative(crossing_path)
        rows.append(row)
    with (root / 'samples.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['id', 'image', 'artery', 'vein'] + (['crossing'] if args.export_crossing else [])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    info = {'count': len(rows), 'crossing_policy': 'independent_mask' if args.export_crossing else 'both',
            'crossing_mask_exported': args.export_crossing,
            'white_policy': args.white_policy, 'pixel_counts': totals,
            'note': 'Mapping choices are not confirmed TFFM author preprocessing.',
            'samples': report}
    (root / 'conversion_report.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Done: {len(rows)} paired samples; binary PNG masks contain 0/255.')
    print(f'Manifest source: {root / "samples.csv"}')
    print('Total source pixel counts:', totals)
    print('Crossings -> both channels; white/uncertain ->', args.white_policy)
    print('Next: run run.py split (or supply your existing split), then run.py train.')


if __name__ == '__main__':
    main()
