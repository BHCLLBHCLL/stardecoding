# -*- coding: utf-8 -*-
"""S2 判别：前 6 s 对齐窗口下，官方（我方网格）vs 自研 n_inner=3 / 5。"""
import io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's6_samemesh', 'transient')
TMAX = 6.0

def load(fn):
    p = os.path.join(W, fn)
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding='utf-8'))
    if 'series' in d:
        t = np.asarray([s['t'] for s in d['series']], float)
        cl = np.asarray([s.get('cl', np.nan) for s in d['series']], float)
    else:
        t = np.asarray(d['t'], float); cl = np.asarray(d['cl'], float)
    ok = np.isfinite(t) & np.isfinite(cl) & (t <= TMAX)
    return t[ok], cl[ok]

def summ(tag, t, cl):
    if t is None or t.size < 4:
        print('%-22s 无数据/样本不足' % tag); return
    n = t.size
    env = [float(np.abs(cl[i*n//3:(i+1)*n//3] - cl[i*n//3:(i+1)*n//3].mean()).max())
           for i in range(3)]
    tt = t[n//2:]; yy = cl[n//2:] - cl[n//2:].mean()
    zc = int((np.diff(np.sign(yy)) != 0).sum())
    print('%-22s n=%3d t=[%.1f, %.1f] 三分窗振幅 %s | 后半段过零 %d | 比值末/首 %.2e'
          % (tag, n, t[0], t[-1], ['%.5f' % v for v in env], zc,
             (env[-1]/env[0]) if env[0] > 0 else float('nan')))

o = load('official_cl_series.json')
summ('official(our mesh)', *o) if o else None
for fn, tag in (('ours_cl_series.json', 'ours ni=3'),
                ('ours_cl_series_ni5.json', 'ours ni=5')):
    d = load(fn)
    if d:
        summ(tag, *d)
    else:
        print('%-22s （尚无数据）' % tag)