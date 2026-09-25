import glob, io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
rows = []
for p in sorted(glob.glob('_screen_*.json')):
    d = json.load(open(p, encoding='utf-8'))
    t = np.asarray(d['t'], float); cl = np.asarray(d['cl'], float)
    ok = np.isfinite(t) & np.isfinite(cl)
    t, cl = t[ok], cl[ok]
    n = cl.size
    if n < 20:
        rows.append((d['tag'], d.get('ni'), d.get('alpha_p'), d.get('conv'), n, '样本不足', None)); continue
    win = max(n // 6, 5)
    env = [float(np.abs(cl[i:i+win] - cl[i:i+win].mean()).max()) for i in range(0, n - win + 1, win)]
    xs = np.arange(len(env)) * win * 0.01
    good = np.array(env) > 1e-7
    sigma = float(np.polyfit(xs[good], np.log(np.array(env)[good]), 1)[0]) if good.sum() >= 2 else None
    rows.append((d['tag'], d.get('ni'), d.get('alpha_p'), d.get('conv'), n,
                 ' '.join('%.4f' % v for v in env[:6]), sigma))
print('%-8s %-4s %-6s %-8s %-5s %-34s %s' % ('tag', 'ni', 'a_p', 'conv', 'n', '包络(前6窗)', 'sigma/s'))
for r in rows:
    print('%-8s %-4s %-6s %-8s %-5d %-34s %s'
          % (r[0], r[1], r[2], r[3], r[4], r[5],
             ('%+.3f' % r[6]) + ('（增长）' if (r[6] or 0) > 0 else '（衰减）') if r[6] is not None else '—'))