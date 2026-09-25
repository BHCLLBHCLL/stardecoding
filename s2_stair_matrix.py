# -*- coding: utf-8 -*-
"""S2 待执行矩阵：阶梯网格（与官方同网格）长窗对照 —— 机器空闲时一条命令跑完。

背景（第八~十三步）：官方在同一张网格上 St=0.1689（σ=+0.062/s），自研 σ=−0.49/s；
短窗筛选无法分辨长时增长率，且外部 OpenFOAM 占满 8 核时算例慢 10–30×。
本脚本按顺序跑若干配置的 20 s 长窗，供空闲窗口一次性执行：
  python s2_stair_matrix.py            # 默认 5 个配置
  python s2_stair_matrix.py 8          # 只跑 ni=8 那一档

piso 必须显式传入。s2_samemesh_ours.py 的缺省是 piso=1，而 α_m=1 与 piso≥1
已在第十八步发散；旧矩阵没传这个参数，首选配置会直接炸掉。
upwind2 用 ni=3：ni=1 的延迟二阶比一阶上风更耗散，不能当长窗的二阶格式。
"""
import io, subprocess, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

STEPS = 2000          # 20 s @ dt=0.01
CONFIGS = [
    # tag, ni, conv, alpha_p, alpha_m, nonorth(0/1), corr_limit, piso
    ("stair_am1noc", 3, "central", 0.3, 1.0, "1", 1.0, 0),  # 不欠松弛 + 满额非正交；piso=0
    ("stair_am1", 3, "central", 0.3, 1.0, "0", 1.0, 0),      # 只放开欠松弛；piso=0
    ("stair_noc", 3, "central", 0.3, 0.7, "1", 1.0, 1),      # 只开修正（α_m=0.7 可与 piso=1 同用）
    ("stair_ni8", 8, "central", 0.3, 0.7, "0", 1.0, 1),      # 步内迭代加码
    ("stair_sou", 3, "upwind2", 0.3, 0.7, "0", 1.0, 0),      # 延迟二阶上风，ni=3 才把高阶项迭代出来
]
only = sys.argv[1] if len(sys.argv) > 1 else None
for tag, ni, conv, ap, am, noc, corr, piso in CONFIGS:
    if only and only not in tag:
        continue
    cmd = [sys.executable, "-u", "s2_samemesh_ours.py", str(STEPS), str(ni),
           tag, conv, str(ap), str(am), str(noc), str(corr), str(piso)]
    print("[matrix] %s" % " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    print("[matrix] %s 退出码 %d" % (tag, rc), flush=True)
print("[matrix] 全部完成；用 python s2_judge.py 出对照")