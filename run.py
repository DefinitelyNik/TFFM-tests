"""TFFM training/evaluation reconstruction. See README for unresolved author details."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from PIL import Image
from data import AVDataset, read_manifest, write_manifest, generate_split
from model import ModelConfig, TFFMSegmentationModel
from paper_losses import make_loss
from metrics import binary_metrics


def save_json(path, value):
    # Strict JSON: represent infinite metric as string rather than invalid Infinity.
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float,np.floating)) and not math.isfinite(x):
            return str(x)
        if isinstance(x, np.generic):
            return x.item()
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False), encoding='utf-8')


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(pretrained, baseline=False):
    cfg = ModelConfig()
    cfg.pretrained = pretrained
    cfg.use_multi_level_tffm = not baseline
    return TFFMSegmentationModel(cfg, num_classes=2)


def fingerprint(rows):
    # Content hash, not just filenames; includes split and sample id.
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r:r['id']):
        h.update((r['id']+'|'+r['split']+'|'+r.get('group','')).encode())
        for k in ('image','artery','vein'):
            with open(r[k], 'rb') as f:
                for block in iter(lambda: f.read(1024*1024), b''):
                    h.update(block)
    return h.hexdigest()


def environment():
    versions = {}
    for pkg in ['torch','torchvision','timm','monai','numpy','scipy','scikit-image','Pillow']:
        versions[pkg] = importlib.metadata.version(pkg)
    return {'python': sys.version, 'packages': versions, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def capture_rng(dataset, generator):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state(), 'loader': generator.get_state(),
             'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             'transforms': []}
    if dataset.transforms:
        # Compose and each Randomizable child use their own numpy RandomState.
        for t in [dataset.transforms] + list(dataset.transforms.transforms):
            state['transforms'].append(t.R.get_state() if hasattr(t,'R') else None)
    return state


def restore_rng(state, dataset, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if dataset.transforms:
        for t, s in zip([dataset.transforms]+list(dataset.transforms.transforms), state['transforms']):
            if s is not None:
                t.R.set_state(s)


def checkpoint_save(path, checkpoint):
    tmp = Path(str(path)+'.tmp')
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def load_trusted(path):
    # Full training state contains numpy RNG objects: only load YOUR trusted checkpoints.
    return torch.load(path, map_location='cpu', weights_only=False)


def train(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:
        raise ValueError('Output contains a checkpoint: use --resume or a new --out')
    rows = read_manifest(args.manifest)
    parts = {s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    if not all(parts.values()):
        raise ValueError('Manifest must contain nonempty train, val and test sets')
    digest = fingerprint(rows)
    ds = AVDataset(parts['train'], not args.no_augmentation, args.seed, args.intensity)
    vs = AVDataset(parts['val'], False, args.seed, args.intensity)
    generator = torch.Generator().manual_seed(args.seed)
    # num_workers=0 intentionally makes online MONAI RNG checkpointable.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        generator=generator, num_workers=0, drop_last=False)
    valid = DataLoader(vs, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if len(ds) % args.batch_size == 1:
        raise ValueError('Last batch size 1 may break encoder BatchNorm: choose a different batch size')
    resume = load_trusted(args.resume) if args.resume else None
    config = {k:v for k,v in vars(args).items() if k not in ('func','resume','out','manifest')}
    if resume:
        if resume['data_sha256'] != digest:
            raise ValueError('Resume dataset/split fingerprint differs')
        old = resume['config']
        for k in config:
            if k not in ('epochs','device') and config[k] != old[k]:
                raise ValueError(f'Resume config differs: {k}')
    model = build_model(args.pretrained and resume is None, args.baseline).to(device)
    criterion = make_loss(args.loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(.9,.999), eps=1e-8)
    best, bad, first = float('inf'), 0, 1
    history = []
    if resume:
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        best,bad,first = resume['best'],resume['bad'],resume['epoch']+1
        history = resume['history']
        # Preserve earlier best when resuming into a new directory.
        source_best = Path(args.resume).parent/'best.pt'
        if source_best.exists() and source_best.resolve() != (out/'best.pt').resolve():
            shutil.copy2(source_best,out/'best.pt')
        restore_rng(resume['rng'],ds,generator)
    write_manifest(rows, out/'manifest.csv')
    save_json(out/'config.json',config)
    save_json(out/'environment.json',environment())
    # Snapshot scripts next to results (including supplied repository loss).
    src = Path(__file__).resolve().parent
    snapshot = out/'source'
    snapshot.mkdir(exist_ok=True)
    for name in ('run.py','model.py','loss.py','paper_losses.py','data.py','metrics.py','requirements.txt'):
        p = src/name
        if p.exists():
            shutil.copy2(p,snapshot/name)
    if bad >= args.patience:
        print('Checkpoint already early-stopped; no additional training.')
        return
    for epoch in range(first, args.epochs+1):
        model.train()
        total,n = 0.,0
        for batch in loader:
            x = batch['image'].to(device)
            y = batch['mask'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits,y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x)
            n += len(x)
        model.eval()
        vt,vn = 0.,0
        dice_values = []
        with torch.inference_mode():
            for batch in valid:
                x,y = batch['image'].to(device), batch['mask'].to(device)
                logits = model(x)
                loss = criterion(logits,y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite validation loss')
                vt += loss.item()*len(x)
                vn += len(x)
                p = logits.sigmoid() >= .5
                tp = (p & (y > .5)).sum((2,3)).float()
                den = p.sum((2,3)) + (y>.5).sum((2,3))
                d = torch.where(den>0,2*tp/den.clamp_min(1),torch.ones_like(tp))
                dice_values.extend(d.mean(1).cpu().tolist())
        value = vt/vn
        improved = value < best
        if improved:
            best,bad = value,0
        else:
            bad += 1
        record = {'epoch':epoch,'train_loss':total/n,'val_loss':value,
                  'val_macro_av_dice':float(np.mean(dice_values)), 'bad_epochs':bad}
        history.append(record)
        print(json.dumps(record),flush=True)
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                 'epoch':epoch,'best':best,'bad':bad,'config':config,
                 'data_sha256':digest,'history':history,'rng':capture_rng(ds,generator)}
        checkpoint_save(out/'last.pt',state)
        if improved:
            checkpoint_save(out/'best.pt',state)
        save_json(out/'history.json',history)
        if bad >= args.patience:
            print(f'Early stopping: {bad} epochs without val loss improvement')
            break
    print('Finished. Evaluate best.pt once on the held-out test split.')


def evaluate(args):
    seed_all(42)
    if not 0 < args.threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    state = load_trusted(args.checkpoint)
    config = state['config']
    all_rows = read_manifest(args.manifest)
    digest = fingerprint(all_rows)
    if not args.external and digest != state['data_sha256']:
        raise ValueError('Dataset/split differs from training. Use --external only for declared external data')
    rows = [r for r in all_rows if r['split']==args.split]
    if not rows:
        raise ValueError('Empty requested split')
    model = build_model(False, config['baseline']).to(args.device)
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    ds = AVDataset(rows, False, intensity=config['intensity'])
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    records = []
    sample_number = 0
    with torch.inference_mode():
        for batch in loader:
            probs = model(batch['image'].to(args.device)).sigmoid().cpu().numpy()
            truths = batch['mask'].numpy() > .5
            for sample_id,p,g in zip(batch['id'],probs,truths):
                pred = p >= args.threshold
                # Combined means OR of A/V binary masks, NOT average A/V Dice.
                triples = [('combined',pred.any(0),g.any(0))]
                if not args.combined_only:
                    triples = [('artery',pred[0],g[0]),('vein',pred[1],g[1])] + triples
                for name, pp, gg in triples:
                    result = binary_metrics(pp,gg,args.skeleton_tolerance,args.junction_tolerance)
                    records.append({'id':sample_id,'class':name,**result})
                if args.save_predictions:
                    folder = out/'predictions'/f'{sample_number:04d}'
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/'id.txt').write_text(sample_id,encoding='utf-8')
                    np.savez_compressed(folder/'probabilities.npz',artery=p[0],vein=p[1])
                    for channel,name in enumerate(('artery','vein')):
                        Image.fromarray(pred[channel].astype(np.uint8)*255).save(folder/f'{name}.png')
                    Image.fromarray(pred.any(0).astype(np.uint8)*255).save(folder/'combined.png')
                sample_number += 1
    with (out/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    summary = {}
    for name in sorted({r['class'] for r in records}):
        rr = [r for r in records if r['class']==name]
        summary[name] = {}
        for key in records[0]:
            if key in ('id','class'):
                continue
            values = np.asarray([r[key] for r in rr],float)
            summary[name][key] = {'mean':float(values.mean()),
                                  'std':float(values.std(ddof=0)) if np.isfinite(values).all() else None,
                                  'nonfinite':int((~np.isfinite(values)).sum())}
    save_json(out/'summary.json', {'n_images':len(rows), 'checkpoint_epoch':state['epoch'],
              'config':config, 'threshold':args.threshold,'data_sha256':digest,
              'external':args.external,'split':args.split,
              'skeleton_tolerance':args.skeleton_tolerance,'junction_tolerance':args.junction_tolerance,
              'aggregation':'per-image macro mean, at 512x512; fraction scores',
              'metrics':summary})
    print(json.dumps(summary,indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('split')
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.set_defaults(func=lambda a: print(generate_split(a.input,a.output,a.seed)))
    p = sub.add_parser('train')
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--loss',choices=['repository','paper'],default='repository')
    p.add_argument('--pretrained',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--baseline',action='store_true',help='Disable TFFM; not an exact reproduction of all paper ablations')
    p.add_argument('--no-augmentation',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=10)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--intensity',choices=['minmax','uint8'],default='minmax')
    p.add_argument('--resume')
    p.set_defaults(func=train)
    p = sub.add_parser('evaluate')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='test')
    p.add_argument('--external',action='store_true')
    p.add_argument('--combined-only',action='store_true')
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--skeleton-tolerance',type=float,default=2.)
    p.add_argument('--junction-tolerance',type=float,default=3.)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
