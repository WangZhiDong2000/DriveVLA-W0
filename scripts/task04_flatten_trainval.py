#!/usr/bin/env python3
"""Flatten navtrain raw metric_cache to metric_cache/trainval/{token}.lzma format."""
import shutil
from pathlib import Path

SRC = Path('/data2/data/navsim/raw_metric_cache/navtrain')
DST = Path('/data2/data/navsim/metric_cache/trainval')
DST.mkdir(parents=True, exist_ok=True)

pkls = list(SRC.rglob('metric_cache.pkl'))
print('Found', len(pkls), 'metric_cache.pkl files')

ok, skip, fail = 0, 0, 0
for pkl in pkls:
    token = pkl.parent.name
    dst = DST / (token + '.lzma')
    if dst.exists():
        skip += 1
        continue
    try:
        shutil.copy2(pkl, dst)
        ok += 1
    except Exception as e:
        print('FAIL', token, ':', e)
        fail += 1

print('Copied:', ok, ', Skipped (exists):', skip, ', Failed:', fail)
print('Total in DST:', len(list(DST.glob('*.lzma'))))
