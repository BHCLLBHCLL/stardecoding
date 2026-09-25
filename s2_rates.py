import io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's6_samemesh', 'transient')
D, U = 0.04, 0.05

def load(fn):
    p = os.path.join(W, fn)
    d = json.load(open(p, encoding='utf-8'))
    if 'series' in d:
        t = np.array([s['t'] for s in d['series']], float)
        cl = np.array([s.get('cl', np.nan) for s in d['series']], float)
    else:
        t = np.array(d['t'], float); cl = np.array(d['cl'], float)
    ok = np.isfinite(t) & np.isfinite(cl)
    return t[ok], cl[ok]

def env_rate(tag, t, cl, win=2.0):
    edges = np.arange(t[0], t[-1] + 1e-9, win)
    env = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() >= 2:
            env.append((0.5*(a+b), float(np.abs(cl[m] - cl[m].mean()).max())))
    env = np.array(env)
    print('%s 包络（%0.1fs 窗）: %s' % (tag, win, ['%.5f@%.1f' % (v, x) for x, v in env]))
    if env.shape[0] >= 3:
        x, y = env[:, 0], env[:, 1]
        ok = y > 1e-7
        p = np.polyfit(x[ok], np.log(y[ok]), 1)
        print('   对数增长率 σ=%+.3f /s（等效数值粘度 ν_num = σ·(D/2π)² = %+.2e；分子粘度 %s）'
              % (p[0], p[0] * (D/(2*np.pi))**2, '1e-05'))
        return p[0]
    return None

t, cl = load('official_cl_series.json')
so = env_rate('官方(我方网格)', t, cl)
for fn, tag in (('ours_cl_series.json', '自研 ni=3'),
                ('ours_cl_series_ni5.json', '自研 ni=5')):
    if os.path.isfile(os.path.join(W, fn)):
        t2, c2 = load(fn)
        env_rate(tag, t2, c2)
if so is not None:
    print('对比：官方 σ=%+.3f/s；自研需额外 ν_num 才能解释衰减 → 见上方各自 σ' % so)