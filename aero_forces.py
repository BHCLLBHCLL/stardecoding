# -*- coding: utf-8 -*-
"""R1-3：气动力系数表面积分（纯 numpy，无 Qt）。

在 P4 `fvm_core.FVM` 单元中心面拓扑上，对**壁面**做应力积分，得到流体作用于
固壁的合力，并按动压归一化得到升/阻力系数 Cd/Cl：

  F_p = Σ_f p_f · n_f · A_f            （压力项，n_f 为流体域外向法线）
  F_v = Σ_f τ_w · t̂_f · A_f            （粘性剪应力，t̂ 沿近壁切向流动方向）
  F   = F_p + F_v
  Cd  = (F·d̂) / (½ ρ_ref U_ref² A_ref)
  Cl  = (F·l̂) / (½ ρ_ref U_ref² A_ref)

符号约定：FVM 边界面法线 n_f 指向**流体域外**（即由 owner 单元指向固壁），
故 `Σ p_f n_f A_f` 正是流体对壁面的压力合力（与动量方程 ∮σ·n dA 一致）：
迎风面 n_f 指向下游，压力贡献正向阻力，符合物理直觉。

粘性项复用 `turbulence.wall_shear`：τ_w = ρ u_τ² = μ|u_t|/d，其中近壁切向
速度 u_t 取 owner 单元速度的切向分量、d 取 owner 质心到壁面质心的距离（一阶
差分），u_τ = sqrt(τ_w/ρ)。剪切方向取近壁切向流动方向（流体对壁面的拖动方向）。

纯 numpy 约束：所有范数走 `solver_run._safe_norm`，不依赖 scipy / np.linalg。
"""
import numpy as np

from solver_run import _safe_norm
from turbulence import wall_shear


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _faces_of(fv, faces):
    """规范化面索引：None 取全部边界面，否则取整型化后的给定面。"""
    if faces is None:
        return np.where(fv.is_boundary)[0]
    return np.asarray(faces, np.int64).ravel()


def _unit(a):
    """单位化向量（零向量抛错，避免静默产生 NaN 结果）。"""
    a = np.asarray(a, float).ravel()
    n = _safe_norm(a)
    if n <= 1e-30:
        raise ValueError("方向向量不可为零向量")
    return a / n


def perpendicular_dir(base, prefer=(0.0, 1.0, 0.0)):
    """返回与 base 正交的单位向量：优先取 prefer 在 base 法平面上的分量。

    prefer 与 base 平行时依次退化为 +z、+x 参考轴。用于升力方向的缺省构造。
    """
    base = _unit(base)
    for ref in (prefer, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
        ref = np.asarray(ref, float)
        w = ref - float(np.dot(ref, base)) * base
        if _safe_norm(w) > 1e-9:
            return w / _safe_norm(w)
    raise ValueError("无法构造与 %s 正交的方向向量" % (base,))


def _face_scalar(fv, x, faces):
    """把标量 / 单元场 / 面场广播到给定面的面值（无插值，取 owner 值）。"""
    arr = np.asarray(x, float)
    if arr.ndim == 0:
        return np.full(len(faces), float(arr), float)
    arr = arr.ravel()
    if arr.shape == (fv.n_cells,):
        return arr[fv.owner[faces]]
    if arr.shape == (fv.n_faces,):
        return arr[faces]
    raise ValueError("物性场维度不匹配：%s（需标量 / 单元 %d / 面 %d）"
                     % (arr.shape, fv.n_cells, fv.n_faces))


# ---------------------------------------------------------------------------
# 分项力积分
# ---------------------------------------------------------------------------
def pressure_force(fv, p, faces=None, p_ref=0.0):
    """压力合力 F_p = Σ_f (p_f - p_ref) n_f A_f（单位：N，缺省全部边界面）。

    p 可为逐单元压力场 (n_cells,) 或逐面压力值 (n_faces,)；面压力缺省取 owner
    单元值。对**封闭壁面**，均匀压力偏置 p_ref 的贡献因 Σ n_f A_f = 0 而自动
    抵消，故 p_ref 仅对非封闭壁面（如仅取的通道侧壁）有意义。
    """
    f = _faces_of(fv, faces)
    if len(f) == 0:
        return np.zeros(3)
    pf = _face_scalar(fv, p, f)
    n = fv.face_normal[f]
    A = fv.face_area[f]
    return np.sum(((pf - p_ref) * A)[:, None] * n, axis=0)


def wall_shear_stress(fv, u, v, w, mu, rho, faces=None):
    """壁面剪应力合力及逐面剪应力/方向。

    近壁切向速度 u_t 取 owner 单元速度的切向分量（扣除法向），壁距 d 取
    owner 质心到面质心距离；τ_w = μ|u_t|/d = ρ u_τ²（经 `turbulence.wall_shear`
    统一），方向 t̂ = u_t/|u_t|（流体对壁面的拖动方向，与第三定律一致）。

    返回 (F_v[3], tau_w[n_faces], t_hat[n_faces,3])。
    """
    f = _faces_of(fv, faces)
    if len(f) == 0:
        return np.zeros(3), np.zeros(0, float), np.zeros((0, 3), float)
    o = fv.owner[f]
    U = np.stack([np.asarray(u, float).ravel()[o],
                  np.asarray(v, float).ravel()[o],
                  np.asarray(w, float).ravel()[o]], axis=1)
    n = fv.face_normal[f]
    un = np.einsum("ij,ij->i", U, n)
    ut = U - un[:, None] * n
    ut_mag = _safe_norm(ut, axis=1)
    d = np.maximum(fv._dL[f], 1e-30)
    mu_f = _face_scalar(fv, mu, f)
    rho_f = _face_scalar(fv, rho, f)
    tau = mu_f * ut_mag / d
    u_tau = np.sqrt(np.maximum(tau / np.maximum(rho_f, 1e-30), 0.0))
    tau = wall_shear(u_tau, rho_f)
    t_hat = np.zeros_like(ut)
    nz = ut_mag > 1e-30
    t_hat[nz] = ut[nz] / ut_mag[nz, None]
    A = fv.face_area[f]
    F = np.sum((tau * A)[:, None] * t_hat, axis=0)
    return F, tau, t_hat


def surface_forces(fv, p, u, v, w, mu, rho, faces=None, p_ref=0.0):
    """壁面总合力（压力 + 粘性）。返回结果字典。"""
    f = _faces_of(fv, faces)
    fp = pressure_force(fv, p, f, p_ref=p_ref)
    fv_force, tau, t_hat = wall_shear_stress(fv, u, v, w, mu, rho, f)
    return {"force": fp + fv_force, "pressure": fp, "viscous": fv_force,
            "tau": tau, "shear_dir": t_hat, "face_area": fv.face_area[f],
            "n_faces": int(len(f))}


# ---------------------------------------------------------------------------
# 力系数
# ---------------------------------------------------------------------------
def force_coefficients(fv, p, u, v, w, mu, rho, faces=None, *,
                       a_ref=1.0, u_ref=1.0, rho_ref=None,
                       drag_dir=(1.0, 0.0, 0.0), lift_dir=None, p_ref=0.0):
    """升/阻力系数：Cd = F·d̂ / q，Cl = F·l̂ / q，q = ½ ρ_ref U_ref² A_ref。

    drag_dir 缺省 +x；lift_dir 缺省取 drag_dir 法平面内偏向 +y 的正交单位向量
    （二维外流 d̂=x 时 l̂=y）。rho_ref 缺省取 rho 的首值（标量或逐单元场皆可）。
    """
    res = surface_forces(fv, p, u, v, w, mu, rho, faces, p_ref=p_ref)
    d_hat = _unit(drag_dir)
    l_hat = _unit(lift_dir) if lift_dir is not None else perpendicular_dir(d_hat)
    if rho_ref is None:
        arr = np.asarray(rho, float).ravel()
        rho_ref = float(arr[0])
    rho_ref = float(rho_ref)
    u_ref = float(u_ref)
    a_ref = float(a_ref)
    q = 0.5 * rho_ref * u_ref * u_ref * a_ref
    if q <= 1e-30:
        raise ValueError("动压归一因子非正：ρ=%g U=%g A=%g" % (rho_ref, u_ref, a_ref))
    F = res["force"]
    res.update({"cd": float(np.dot(F, d_hat) / q),
                "cl": float(np.dot(F, l_hat) / q),
                "q": float(q), "rho_ref": rho_ref, "u_ref": u_ref,
                "a_ref": a_ref, "drag_dir": d_hat, "lift_dir": l_hat})
    return res
