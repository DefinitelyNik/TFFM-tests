"""Re-evaluate an EXISTING trusted best.pt without retraining, including CIs.
Edit CHECKPOINT below, then Run / %run start_evaluation.py. No CLI flags needed.
Keep this file next to run.py and interval_metrics.py.
"""
from pathlib import Path
from datetime import datetime
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent
# REQUIRED: explicit checkpoint; never silently choose the most recent model.
# Example: PROJECT_DIR / 'runs' / 'tffm_repository_...' / 'best.pt'
CHECKPOINT = None
MANIFEST = PROJECT_DIR / 'datasets' / 'fundus_avseg' / 'manifest.csv'
BATCH_SIZE = 1
BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 42
THRESHOLD = 0.5
SAVE_PREDICTIONS = False


def main():
    import torch
    if CHECKPOINT is None:
        raise ValueError('Set CHECKPOINT to YOUR trusted best.pt in start_evaluation.py first.')
    checkpoint = Path(CHECKPOINT)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_DIR / checkpoint
    checkpoint = checkpoint.resolve()
    manifest = Path(MANIFEST)
    if not manifest.is_absolute():
        manifest = PROJECT_DIR / manifest
    manifest = manifest.resolve()
    for path in (checkpoint, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    from start_training import make_zip
    out = checkpoint.parent / ('test_intervals_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    arguments = [
        sys.executable, '-u', str(PROJECT_DIR / 'run.py'), 'evaluate',
        '--checkpoint', str(checkpoint), '--manifest', str(manifest),
        '--out', str(out), '--split', 'test', '--batch-size', str(BATCH_SIZE),
        '--device', 'cuda' if torch.cuda.is_available() else 'cpu',
        '--threshold', str(THRESHOLD), '--bootstrap-samples', str(BOOTSTRAP_SAMPLES),
        '--confidence', str(CONFIDENCE), '--bootstrap-seed', str(BOOTSTRAP_SEED),
    ]
    if SAVE_PREDICTIONS:
        arguments.append('--save-predictions')
    # Fingerprint checks in run.py remain ENABLED. No automatic --external.
    subprocess.run(arguments, cwd=str(PROJECT_DIR), check=True)
    # Archive reports ONLY, without duplicating old checkpoint archives.
    archives = PROJECT_DIR / 'runs' / 'archives'
    archives.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(out)
        if 'predictions' in relative.parts or path.suffix.lower() in ('.pt', '.pth', '.ckpt', '.tmp', '.zip'):
            continue
        items.append((path, (Path(checkpoint.parent.name) / out.name / relative).as_posix()))
    make_zip(archives / f'{checkpoint.parent.name}_{out.name}_metrics_history.zip', items)
    print('Evaluation only: training/checkpoints unchanged. Report archive contains no training history unless already in the report directory.')
    print(f'Reports: {out}')


if __name__ == '__main__':
    main()
