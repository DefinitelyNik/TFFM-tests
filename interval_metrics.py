"""Percentile cluster-bootstrap intervals for a FIXED model on a held-out split.
No cross-validation, retraining, pixel bootstrap or inference on missing classes.
Only numpy and the standard library are used here.
"""
import csv
import json
from pathlib import Path
import numpy as np


NAME_MAP = {
    'artery': 'arteries', 'vein': 'veins',
    'combined': 'vessel_union', 'crossing': 'crossings', 'background': 'background',
}
METRIC_MAP = {
    'pred_components': 'components_pred', 'gt_components': 'components_gt',
    'hd95': 'hd95_px',
}
FIELDS = ['metric_name', 'average', 'lower_bound', 'upper_bound',
          'valid_images', 'undefined_images', 'groups', 'bootstrap_samples', 'scope']


def write_interval_report(records, rows, output, samples=2000, confidence=.95,
                          seed=42, context=None):
    """Resample entire patient/groups with replacement; average over images.

    Estimand: mean per-image score conditional on that metric being finite.
    Each replicate contains ALL images of each sampled group, including repeats.
    Thus this is not the unweighted mean of patient means for unequal group sizes.
    Nonfinite values are not changed to zero or to arbitrary finite distances.
    The numbers of omitted values and all bootstrap settings are saved.
    """
    if samples < 100 or not 0 < confidence < 1:
        raise ValueError('Need samples >= 100 and confidence between 0 and 1')
    if not rows or not records:
        raise ValueError('Empty interval report input')
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image ids in interval report')
    given_groups = [str(r.get('group', '')).strip() for r in rows]
    if any(given_groups) and not all(given_groups):
        raise ValueError('For cluster bootstrap, provide group for ALL images or NONE')
    group_values = given_groups if all(given_groups) else ids
    group_names = sorted(set(group_values))
    group_index = {g: i for i, g in enumerate(group_names)}
    image_groups = np.asarray([group_index[g] for g in group_values], dtype=int)
    n_groups = len(group_names)
    index = {sample_id: i for i, sample_id in enumerate(ids)}
    columns = {}
    seen = set()
    for record in records:
        sample_id, class_name = record['id'], record['class']
        if sample_id not in index or (sample_id, class_name) in seen:
            raise ValueError('Unknown image or repeated image/class record')
        seen.add((sample_id, class_name))
        prefix = NAME_MAP.get(class_name, class_name)
        for metric, value in record.items():
            if metric in ('id', 'class'):
                continue
            name = prefix + '_' + METRIC_MAP.get(metric, metric)
            columns.setdefault(name, np.full(len(ids), np.nan))[index[sample_id]] = float(value)
    # Average A/V PER IMAGE first, then bootstrap. Not the union score and
    # not the 4-foreground-class macro in a five-class multiclass evaluation.
    for metric in ('dice', 'iou', 'precision', 'recall'):
        a, v = 'arteries_' + metric, 'veins_' + metric
        if a in columns and v in columns:
            columns['macro_av_' + metric] = (columns[a] + columns[v]) / 2

    rng = np.random.default_rng(seed)
    # Same group draws for every metric. Batched to bound memory usage.
    boot = {name: np.full(samples, np.nan) for name in columns}
    stats = {}
    for name, values in columns.items():
        finite = np.isfinite(values)
        sums = np.bincount(image_groups, weights=np.where(finite, values, 0.),
                           minlength=n_groups)
        counts = np.bincount(image_groups, weights=finite.astype(float),
                             minlength=n_groups)
        stats[name] = (sums, counts, finite)
    if n_groups >= 2:
        for start in range(0, samples, 128):
            stop = min(start + 128, samples)
            draws = rng.integers(0, n_groups, size=(stop - start, n_groups))
            for name, (sums, counts, _) in stats.items():
                numerator = sums[draws].sum(axis=1)
                denominator = counts[draws].sum(axis=1)
                boot[name][start:stop] = np.divide(
                    numerator, denominator, out=np.full(stop - start, np.nan),
                    where=denominator > 0,
                )
    alpha = (1 - confidence) / 2
    result, per_metric = [], {}
    scope = 'conditional_mean_per_image_holdout_cluster_bootstrap'
    for name in sorted(columns):
        values = columns[name]
        finite = stats[name][2]
        draws = boot[name][np.isfinite(boot[name])]
        bounds = np.quantile(draws, [alpha, 1 - alpha]) if len(draws) else (None, None)
        result.append({
            'metric_name': name,
            'average': float(values[finite].mean()) if finite.any() else None,
            'lower_bound': float(bounds[0]) if bounds[0] is not None else None,
            'upper_bound': float(bounds[1]) if bounds[1] is not None else None,
            'valid_images': int(finite.sum()),
            'undefined_images': int((~finite).sum()),
            'groups': n_groups,
            'bootstrap_samples': samples,
            'scope': scope,
        })
        per_metric[name] = {
            'valid_bootstrap_samples': int(len(draws)),
            'groups_with_finite_values': int(np.unique(image_groups[finite]).size),
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'interval_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(result)
    # Wide format for independent analysis / paired comparisons on shared ids.
    with (output / 'interval_per_image.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'group', *sorted(columns)])
        writer.writeheader()
        for i, sample_id in enumerate(ids):
            record = {'id': sample_id, 'group': group_values[i]}
            record.update({name: float(v[i]) if np.isfinite(v[i]) else ''
                           for name, v in columns.items()})
            writer.writerow(record)
    metadata = {
        'confidence_level': confidence, 'method': 'percentile cluster bootstrap',
        'seed': seed, 'bootstrap_samples': samples, 'n_images': len(ids),
        'groups': n_groups, 'group_source': 'group column' if all(given_groups) else 'image id',
        'scope': scope, 'resampling_unit': 'whole group; image when no group supplied',
        'estimand': 'image-weighted mean conditional on finite metric value',
        'per_metric': per_metric, 'context': context or {},
        'label_semantics': {
            'arteries': 'binary artery target AS STORED; normally red OR green',
            'veins': 'binary vein target AS STORED; normally blue OR green',
            'vessel_union': 'OR of the two binary channels',
            'crossings': 'intersection of the two native binary predictions; target is the independent crossing mask when present, otherwise artery AND vein target',
            'background': 'complement of vessel_union, not necessarily RGB black',
            'macro_av': 'mean of artery and vein metrics for each image',
        },
        'not_available': {
            'uncertain_vessels': 'No uncertain-vessel output/target; not assigned a fabricated score',
            'macro_dice_all_and_foreground_macro_dice': 'Five-class macro is not identifiable from native binary targets',
        },
        'warnings': [
            'Intervals condition on this fitted model and split; no training/seed/CV uncertainty.',
            'No patient ids supplied: independence of different images is an assumption.'
            if not all(given_groups) else 'All images of a sampled group are resampled together.',
            'Nonfinite values are omitted only for the conditional report; see summary.json for raw results.',
            'A finite-only HD95 mean can hide empty-mask failures; inspect undefined_images.',
            'The empty-mask and HD95 conventions are those of metrics.py, not verified UNET++ WCD code.',
            'Very small held-out samples yield unstable intervals; bootstrap cannot add independent data.',
        ],
    }
    if n_groups < 2:
        metadata['warnings'].append('Fewer than two groups: interval bounds intentionally left empty.')
    (output / 'interval_metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8'
    )
    print(f'Interval report: {output / "interval_metrics.csv"}', flush=True)
    return result
