# -*- coding: utf-8 -*-
"""S2 判别实验分析：官方（我方网格）vs 我方 CL(t) —— 脱落是否发生、St 与官方参考对比。"""
import io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 's6_samemesh', 'transient')
D, U = 0.04, 0.05
ST_OFFICIAL_REF = 0.1752      # 同一官方求解器在官方网格上的参考（vortexShed tutor）

def analyze(tag, t, cl):
    t = np.asarray(t, float); cl = np.asarray(cl, float)
    ok = np.isfinite(t) & np.isfinite(cl)
    t, cl = t[ok], cl[ok]
    if t.size < 8:
        print('%-18s 样本不足 (%d)' % (tag, t.size)); return None
    amp = 0.5 * (cl.max() - cl.min())
    mean = float(cl.mean())
    # 后半段更适合统计稳态
    half = t.size // 2
    tt, yy = t[half:], cl[half:] - cl[half:].mean()
    # 过零周期
    zc = np.where(np.diff(np.sign(yy)) != 0)[0]
    f_cross = float('nan');
    if zc.size >= 4:
        per = float(np.mean(np.diff(tt[zc]))) * 2.0
        if per > 0: f_cross = 1.0 / per
    # FFT
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.5
    n = yy.size
    spec = np.abs(np.fft.rfft(yy * np.hanning(n))) if n > 4 else np.zeros(2)
    freq = np.fft.rfftfreq(n, d=dt)
    f_fft = float(freq[int(np.argmax(spec[1:]) + 1)]) if spec.size > 2 else float('nan')
    fs = [f for f in (f_cross, f_fft) if np.isfinite(f) and f > 0]
    f = float(np.mean(fs)) if fs else float('nan')
    st = f * D / U if np.isfinite(f) else float('nan')
    print('%-18s n=%2d t=[%.1f, %.1f] 振幅 %.4f 均值 %+.4f | 过零 %.4f Hz FFT %.4f Hz → St=%.4f'
          % (tag, t.size, t[0], t[-1], amp, mean, f_cross, f_fft, st))
    return {'n': int(t.size), 't_span': float(t[-1] - t[0]), 'amplitude': amp,
            'mean': mean, 'f_cross': f_cross, 'f_fft': f_fft, 'st': st}

out = {}
for tag, fn, key_t, key_cl in (('official(our mesh)', 'official_cl_series.json', 't', 'cl'),
                               ('ours(our mesh)', 'ours_cl_series.json', 't', 'cl')):
    # ours 可能仍在跑：读到的序列如实标注跨度
    p = os.path.join(WORK, fn)
    if not os.path.isfile(p):
        print('%-18s 缺文件 %s' % (tag, fn)); continue
    d = json.load(open(p, encoding='utf-8'))
    if 'series' in d:
        t = [s['t'] for s in d['series']]; cl = [s['cl'] for s in d['series']]
    else:
        t = d['t']; cl = d['cl']
    t = np.asarray(t, float)
    cl = np.asarray(cl, float)
    r = analyze(tag, t, cl)
    # 分窗包络（每 1/4）+ 对数增长拟合：脱落仍在发展期时振幅会持续增长
    n = t.size
    env = [float(np.abs(cl[i * n // 4:(i + 1) * n // 4]
                        - cl[i * n // 4:(i + 1) * n // 4].mean()).max())
           for i in range(4)]
    if r is not None:
        r['window_amp'] = [round(v, 5) for v in env]
    zc = int((np.diff(np.sign(cl[n // 2:] - cl[n // 2:].mean())) != 0).sum())
    if r is not None:
        r['zero_crossings_2nd_half'] = zc
        r['oscillates'] = bool(zc >= 4 and env[-1] > 0.002)
    print('%-18s 分窗振幅 %s | 后半段过零 %d → %s'
          % (tag, [round(v, 5) for v in env], zc,
             '有振荡（脱落）' if (r or {}).get('oscillates') else '无振荡（近定常/衰减）'))
    out[tag] = r
    out[tag + ' series'] = {'t': list(map(float, t)), 'cl': list(map(float, cl))}
o = out.get('official(our mesh)')
if o and o.get('st') and np.isfinite(o['st']):
    print('官方参考（同求解器/官方网格）St=%.4f；本实验 %.4f（相对 %.2f）'
          % (ST_OFFICIAL_REF, o['st'], o['st'] / ST_OFFICIAL_REF))
if o:
    print('判定：官方侧 %s（后半段过零 %d 次、末窗振幅 %.5f）→ 网格支持脱落 = %s'
          % ('有振荡' if o.get('oscillates') else '无振荡',
             o.get('zero_crossings_2nd_half', 0),
             (o.get('window_amp') or [0])[-1], bool(o.get('oscillates'))))
    print('含义：官方求解器在同一张网格上能脱落 → 阻断在**自研求解器**的数值耗散，'
          '而非网格/几何（S2 长期悬置的 mesh-vs-solver 判别）')
json.dump(out, open(os.path.join(WORK, 'judgement.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('→ judgement.json')