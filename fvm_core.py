# -*- coding: utf-8 -*-
"""P 波 P4：有限体积离散核心（纯 numpy，occ / scdm 两环境皆可用）。

面向**四面体**体网格 (vertices[N,3], cells[M,4]) 的中心式有限体积离散，
提供 P5 压力基求解器所需的离散算子：

  1) cell-face 拓扑构建：从四面体单元导出内部面（两单元共享）与边界面
     （仅一单元），给每个面配 质心 / 面积 / 外法向 / owner-neighbor 映射。
  2) 单元几何：体积 + 质心（四面体质心 = 四顶点均值）。
  3) 梯度：Green-Gauss（面值线性插值）与 Least-Squares（法方程 3x3，
     用显式伴随式求解，不依赖 LAPACK/BLAS，规避 occ 环境崩溃）。
  4) 面插值：线性反距离权重 φ_f = (d_R φ_L + d_L φ_R)/(d_L + d_R)。
  5) 限制器：Barth-Jespersen（TVD，线性场精确 = 1，近极值 <1）。
  6) 通量格式：扩散中心差分 + 对流 上风 / 中心 格式（面量纲标量）。

纯 numpy 约束：scdm 环境无 scipy，故不使用 scipy.sparse；所有范数统一走
`solver_run._safe_norm`（sqrt(sum(x*x))），避免 occ 环境 np.linalg 触发
Windows fatal exception (0xc06d007f)。

验收核心（P4 行）：FVM 离散核心 —— 梯度(Gauss/LSQ)、限制器、通量格式 向量化可用。
"""
import numpy as np

from solver_run import _safe_norm


# ---------------------------------------------------------------------------
# 单元几何（体积 / 质心）
# ---------------------------------------------------------------------------
def _tet_volume(V, cells):
    """四面体带号体积（正 = 右手定向）。公式：det(b-a, c-a, d-a)/6。"""
    a = V[cells[:, 0]]
    b = V[cells[:, 1]]
    c = V[cells[:, 2]]
    d = V[cells[:, 3]]
    e1 = b - a
    e2 = c - a
    e3 = d - a
    return (e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
            - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
            + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])) / 6.0


def _cell_geometry(V, C):
    """单元体积（绝对值）与质心。返回 (vol[M], centroid[M,3])。"""
    vol = _tet_volume(V, C)
    return np.abs(vol), V[C].mean(axis=1)


def _solve3x3_vec(A, b):
    """批量解 3x3 线性方程 A x = b（显式伴随式，逐单元向量化）。

    A: (M,3,3)，b: (M,3)，返回 x: (M,3)。行列式近零的单元返回 0 梯度。
    """
    a11 = A[:, 0, 0]; a12 = A[:, 0, 1]; a13 = A[:, 0, 2]
    a21 = A[:, 1, 0]; a22 = A[:, 1, 1]; a23 = A[:, 1, 2]
    a31 = A[:, 2, 0]; a32 = A[:, 2, 1]; a33 = A[:, 2, 2]
    det = (a11 * (a22 * a33 - a23 * a32)
           - a12 * (a21 * a33 - a23 * a31)
           + a13 * (a21 * a32 - a22 * a31))
    inv00 = a22 * a33 - a23 * a32
    inv01 = a13 * a32 - a12 * a33
    inv02 = a12 * a23 - a13 * a22
    inv10 = a23 * a31 - a21 * a33
    inv11 = a11 * a33 - a13 * a31
    inv12 = a13 * a21 - a11 * a23
    inv20 = a21 * a32 - a22 * a31
    inv21 = a12 * a31 - a11 * a32
    inv22 = a11 * a22 - a12 * a21
    safe = np.where(np.abs(det) > 1e-14, det, 1.0)
    den = np.where(np.abs(det) > 1e-14, 1.0, 0.0)
    b0 = b[:, 0]; b1 = b[:, 1]; b2 = b[:, 2]
    x0 = (inv00 * b0 + inv10 * b1 + inv20 * b2) / safe * den
    x1 = (inv01 * b0 + inv11 * b1 + inv21 * b2) / safe * den
    x2 = (inv02 * b0 + inv12 * b1 + inv22 * b2) / safe * den
    return np.stack([x0, x1, x2], axis=1)


# ---------------------------------------------------------------------------
# 有限体积离散核心
# ---------------------------------------------------------------------------
class FVM:
    """四面体网格上的中心式有限体积离散核心（纯 numpy）。

    构建面拓扑后提供梯度 / 面插值 / 限制器 / 通量格式算子。face 数组索引约定：
      - owner[o, f]、neighbor[nb, f]，nb = -1 表示边界面；
      - face_normal 指向 owner → neighbor（即 owner 的外向法线）；
      - 内部面两单元共享且方向相反，边界面法线指向体网格外部。
    """

    def __init__(self, vertices, cells):
        self.vertices = np.asarray(vertices, float)
        self.cells = np.asarray(cells, np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("FVM 需要 Nx3 顶点坐标")
        if (self.cells.ndim != 2 or self.cells.shape[1] != 4):
            raise ValueError("FVM 需要 Mx4 四面体单元")
        if len(self.vertices) < 4 or len(self.cells) < 1:
            raise ValueError("FVM 网格过小（顶点<4 或单元<1）")
        self.n_cells = len(self.cells)
        self.n_vertices = len(self.vertices)
        self.volumes, self.centroids = _cell_geometry(self.vertices, self.cells)
        self._build_faces()

    # -- 面拓扑 ----------------------------------------------------------
    def _build_faces(self):
        V = self.vertices
        C = self.cells
        M = self.n_cells
        locs = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
        F = np.zeros((4 * M, 3), np.int64)
        cell_of = np.zeros(4 * M, np.int64)
        opp = np.zeros(4 * M, np.int64)
        for k, (a, b, c) in enumerate(locs):
            F[k * M:(k + 1) * M] = C[:, (a, b, c)]
            cell_of[k * M:(k + 1) * M] = np.arange(M)
            opp[k * M:(k + 1) * M] = C[:, k]
        p0 = V[F[:, 0]]
        p1 = V[F[:, 1]]
        p2 = V[F[:, 2]]
        e0 = p1 - p0
        e1 = p2 - p0
        raw_n = np.cross(e0, e1)
        raw_len = _safe_norm(raw_n, axis=1)
        area = 0.5 * raw_len
        unit = np.zeros_like(raw_n)
        good = raw_len > 1e-300
        unit[good] = raw_n[good] / raw_len[good, None]
        centroid = (p0 + p1 + p2) / 3.0
        dot = np.sum(raw_n * (centroid - V[opp]), axis=1)
        unit[dot < 0.0] *= -1.0
        N = self.n_vertices
        kv = np.sort(F, axis=1)
        key = (kv[:, 0].astype(np.int64) * N + kv[:, 1]) * N + kv[:, 2]
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        g_start = np.concatenate(([0], np.where(np.diff(sorted_key) != 0)[0] + 1))
        g_end = np.concatenate((g_start[1:], [len(order)]))
        nf = len(g_start)
        owner = np.full(nf, -1, np.int64)
        neighbor = np.full(nf, -1, np.int64)
        fnorm = np.zeros((nf, 3))
        farea = np.zeros(nf)
        fcent = np.zeros((nf, 3))
        fverts = np.zeros((nf, 3), np.int64)
        isbnd = np.zeros(nf, bool)
        for gi in range(nf):
            rows = order[g_start[gi]:g_end[gi]]
            if len(rows) == 1:
                c0 = int(cell_of[rows[0]])
                owner[gi] = c0
                neighbor[gi] = -1
                isbnd[gi] = True
                fnorm[gi] = unit[rows[0]]
                farea[gi] = area[rows[0]]
                fcent[gi] = centroid[rows[0]]
                fverts[gi] = F[rows[0]]
            elif len(rows) == 2:
                c0 = int(cell_of[rows[0]])
                c1 = int(cell_of[rows[1]])
                if c0 == c1:
                    continue
                if c0 < c1:
                    o, nb, row_o = c0, c1, rows[0]
                else:
                    o, nb, row_o = c1, c0, rows[1]
                owner[gi] = o
                neighbor[gi] = nb
                fnorm[gi] = unit[row_o]
                farea[gi] = area[row_o]
                fcent[gi] = centroid[row_o]
                fverts[gi] = F[row_o]
        self.face_vertices = fverts
        self.face_normal = fnorm
        self.face_area = farea
        self.face_centroid = fcent
        self.owner = owner
        self.neighbor = neighbor
        self.is_boundary = isbnd
        self.n_faces = nf
        self.n_interior_faces = int((~isbnd).sum())
        self.n_boundary_faces = int(isbnd.sum())
        self._dL = np.zeros(nf, float)
        self._dR = np.zeros(nf, float)
        self._d_n = np.zeros(nf, float)
        self._compute_face_distances()

    def _compute_face_distances(self):
        fo = self.owner
        nb = self.neighbor
        is_int = nb >= 0
        nbr = np.where(is_int, nb, 0)
        cell_safe = np.where(is_int, nbr, fo)
        dL = _safe_norm(self.face_centroid - self.centroids[fo], axis=1)
        dR = _safe_norm(self.face_centroid - self.centroids[cell_safe], axis=1)
        self._dL = dL
        self._dR = dR
        delta = self.centroids[cell_safe] - self.centroids[fo]
        n_dot = np.sum(delta * self.face_normal, axis=1)
        self._d_n = np.where(np.abs(n_dot) > 1e-14, np.abs(n_dot),
                             np.maximum(dL + dR, 1e-12))

    # -- 校验 ------------------------------------------------------------
    def _check_phi(self, phi):
        phi = np.asarray(phi, float)
        if phi.shape != (self.n_cells,):
            raise ValueError("FVM 场量与单元数不匹配：%s vs %d"
                             % (phi.shape, self.n_cells))
        return phi

    def _check_boundary(self, boundary):
        if boundary is None:
            return None
        b = np.asarray(boundary, float)
        b = b.ravel()
        if b.shape != (self.n_faces,):
            raise ValueError("FVM 边界场量与面数不匹配：%s vs %d"
                             % (b.shape, self.n_faces))
        return b

    # -- 梯度 ------------------------------------------------------------
    def grad_gauss(self, phi, boundary=None, recon=None):
        """Green-Gauss 梯度：∇φ_c = (1/V) Σ_f φ_f n_f A_f。

        φ_f 内部面默认取线性平均（0.5(φ_L+φ_R)，对歪斜网格仅一阶/不收敛）；
        传入 `recon`（如 LSQ 梯度数组 (n_cells,3)）则用二阶重构面值
        φ_f = 0.5(φ_L+∇φ_L·(x_f-x_L) + φ_R+∇φ_R·(x_f-x_R))，线性场可复原到机器精度。
        边界面取 boundary[f]（缺省 = 单元值，零梯度外推）。
        """
        phi = self._check_phi(phi)
        boundary = self._check_boundary(boundary)
        is_int = self.neighbor >= 0
        nbr = np.where(is_int, self.neighbor, 0)
        phi_f = np.empty(self.n_faces, float)
        if recon is None:
            phi_f[is_int] = 0.5 * (phi[self.owner[is_int]] + phi[nbr[is_int]])
        else:
            recon = np.asarray(recon, float)
            if recon.shape != (self.n_cells, 3):
                raise ValueError("FVM 重构梯度需 (n_cells,3)")
            xf = self.face_centroid
            dL = np.einsum("ij,ij->i", recon[self.owner], xf - self.centroids[self.owner])
            dR = np.einsum("ij,ij->i", recon[nbr], xf - self.centroids[nbr])
            phi_f[is_int] = 0.5 * (phi[self.owner[is_int]] + dL[is_int]
                                   + phi[nbr[is_int]] + dR[is_int])
        if boundary is not None:
            phi_f[~is_int] = boundary[~is_int]
        else:
            phi_f[~is_int] = phi[self.owner[~is_int]]
        flux = phi_f[:, None] * self.face_normal * self.face_area[:, None]
        acc = np.zeros((self.n_cells, 3))
        np.add.at(acc, self.owner, flux)
        np.add.at(acc, nbr[is_int], -flux[is_int])
        vol = np.where(self.volumes > 1e-14, self.volumes, 1.0)
        return acc / vol[:, None]

    def grad_lsq(self, phi, boundary=None):
        """最小二乘梯度：拟合 Δφ ≈ grad · Δx（相邻单元 + 边界面样本）。

        样本集：每个内部面为两侧单元各贡献一次（Δx = x_nb - x_o，Δφ = φ_nb - φ_o，
        权重 1/|Δx|）；每个边界面为 owner 贡献一次（Δx = x_f - x_o，
        Δφ = boundary[f] - φ_o，缺省外推 Δφ=0）。边界面样本可打破边界单元
        相邻质心共面导致的法方程奇异，线性场复原到机器精度。
        """
        phi = self._check_phi(phi)
        boundary = self._check_boundary(boundary)
        is_int = self.neighbor >= 0
        o = self.owner[is_int]
        nb = self.neighbor[is_int]
        r_i = self.centroids[nb] - self.centroids[o]
        dq_i = phi[nb] - phi[o]
        # 内部面：owner / neighbor 双侧样本（同一 r, dq）
        R_int = np.concatenate([r_i, r_i])
        Q_int = np.concatenate([dq_i, dq_i])
        S_int = np.concatenate([o, nb])
        # 边界面：仅 owner 样本
        bo = self.owner[~is_int]
        r_b = self.face_centroid[~is_int] - self.centroids[bo]
        if boundary is not None:
            q_b = boundary[~is_int] - phi[bo]
        else:
            q_b = np.zeros(len(bo))
        R = np.concatenate([R_int, r_b])
        Q = np.concatenate([Q_int, q_b])
        S = np.concatenate([S_int, bo])
        d = _safe_norm(R, axis=1)
        w = 1.0 / np.maximum(d, 1e-12)
        G = np.zeros((self.n_cells, 3, 3))
        rhs = np.zeros((self.n_cells, 3))
        wr = w[:, None] * R
        for i in range(3):
            for j in range(3):
                np.add.at(G[:, i, j], S, wr[:, i] * R[:, j])
        np.add.at(rhs, S, (w * Q)[:, None] * R)
        return _solve3x3_vec(G, rhs)

    # -- 面插值 ----------------------------------------------------------
    def face_value(self, phi, boundary=None):
        """面值线性插值：φ_f = (d_R φ_L + d_L φ_R)/(d_L + d_R)。

        内部面用反距离权重；边界面 = boundary[f]（缺省 = 单元值）。
        """
        phi = self._check_phi(phi)
        boundary = self._check_boundary(boundary)
        is_int = self.neighbor >= 0
        nbr = np.where(is_int, self.neighbor, 0)
        den = np.maximum(self._dL + self._dR, 1e-12)
        phi_f = np.empty(self.n_faces, float)
        phi_f[is_int] = ((self._dR[is_int] * phi[self.owner[is_int]]
                          + self._dL[is_int] * phi[nbr[is_int]]) / den[is_int])
        if boundary is not None:
            phi_f[~is_int] = boundary[~is_int]
        else:
            phi_f[~is_int] = phi[self.owner[~is_int]]
        return phi_f

    # -- 限制器 ----------------------------------------------------------
    def limiter(self, phi, grad, kind="barth"):
        """Barth-Jespersen TVD 限制器（逐单元标量，∈[0,1]）。

        对每个内部面，单元在邻居处重构 φ_recon = φ_c + grad·(x_nb - x_c)，
        取其与局部 φ_max/φ_min 比值的最小值。线性场精确返回 1，近极值 <1。
        """
        phi = self._check_phi(phi)
        grad = np.asarray(grad, float)
        if grad.shape != (self.n_cells, 3):
            raise ValueError("FVM 梯度形状需 (n_cells,3)")
        is_int = self.neighbor >= 0
        o = self.owner[is_int]
        nb = self.neighbor[is_int]
        pmin = phi.copy()
        pmax = phi.copy()
        np.minimum.at(pmin, o, phi[nb])
        np.minimum.at(pmin, nb, phi[o])
        np.maximum.at(pmax, o, phi[nb])
        np.maximum.at(pmax, nb, phi[o])
        r = self.centroids[nb] - self.centroids[o]
        delta_o = np.einsum("ij,ij->i", grad[o], r)
        delta_nb = np.einsum("ij,ij->i", grad[nb], -r)
        lim = np.ones(self.n_cells, float)
        np.minimum.at(lim, o, _barth_ratio(delta_o, phi[o], pmin[o], pmax[o]))
        np.minimum.at(lim, nb, _barth_ratio(delta_nb, phi[nb], pmin[nb], pmax[nb]))
        return np.clip(lim, 0.0, 1.0)

    # -- 通量格式 --------------------------------------------------------
    def diffusion_flux(self, phi, gamma=1.0, boundary=None):
        """扩散中心差分面通量（正 = owner → neighbor）。

        F = -gamma (φ_nb - φ_owner)/d_n * A；边界面用 boundary[f]，缺省为
        零梯度（F=0 天然无通量）。
        """
        phi = self._check_phi(phi)
        boundary = self._check_boundary(boundary)
        g = float(gamma)
        is_int = self.neighbor >= 0
        nbr = np.where(is_int, self.neighbor, 0)
        flux = np.zeros(self.n_faces, float)
        flux[is_int] = -g * (phi[nbr[is_int]] - phi[self.owner[is_int]]) \
            / self._d_n[is_int] * self.face_area[is_int]
        if boundary is not None:
            flux[~is_int] = -g * (boundary[~is_int] - phi[self.owner[~is_int]]) \
                / self._d_n[~is_int] * self.face_area[~is_int]
        else:
            flux[~is_int] = 0.0
        return flux

    def convection_flux_upwind(self, mdot, phi, boundary=None):
        """一阶上风对流面通量：F = mdot * φ_upwind（mdot 为面质量通量）。

        内部面取流动上游侧单元值；边界面 mdot>=0 外流取单元值，mdot<0 入流
        取 boundary[f]（缺省 = 单元值）。
        """
        phi = self._check_phi(phi)
        boundary = self._check_boundary(boundary)
        mdot = np.asarray(mdot, float).ravel()
        if mdot.shape != (self.n_faces,):
            raise ValueError("FVM 面质量通量需 (n_faces,)")
        is_int = self.neighbor >= 0
        # 边界面回退到 owner 单元值（无边界值时作外推），内部面取对侧单元
        nbr = np.where(is_int, self.neighbor, self.owner)
        up = np.where(mdot >= 0.0, phi[self.owner], phi[nbr])
        flux = mdot * up
        if boundary is not None:
            inflow = (~is_int) & (mdot < 0.0)
            flux[inflow] = mdot[inflow] * boundary[inflow]
        return flux

    def convection_flux_central(self, mdot, phi, boundary=None):
        """中心对流面通量：F = mdot * φ_f（φ_f 为线性插值面值）。"""
        phi = self._check_phi(phi)
        boundary = self._check_boundary(boundary)
        mdot = np.asarray(mdot, float).ravel()
        if mdot.shape != (self.n_faces,):
            raise ValueError("FVM 面质量通量需 (n_faces,)")
        return mdot * self.face_value(phi, boundary=boundary)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _barth_ratio(delta, phi_c, pmin, pmax):
    """Barth-Jespersen 单面比值（未裁剪）。"""
    ratio = np.ones_like(delta)
    eps = 1e-12
    pos = delta > eps
    neg = delta < -eps
    ratio[pos] = (pmax[pos] - phi_c[pos]) / delta[pos]
    ratio[neg] = (pmin[neg] - phi_c[neg]) / delta[neg]
    return ratio


# ---------------------------------------------------------------------------
# 一致性网格基元（FVM 收敛 / 守恒验证用）
# ---------------------------------------------------------------------------
def cube_tet_mesh(nx=2, ny=None, nz=None):
    """单位立方体结构化四面体网格：每 hex **6-tet Kuhn 分解**（体积精确填满）。

    与 `solver_run.demo_mesh`（5-tet 扇形、绕主对角线留缝、体积和 5/6）不同，
    本函数采用 Kuhn/Freudenthal 三角剖分：6 个四面体绕主对角线 0–6 呈环形
    共享边，体积和恰好为 1.0，且在结构化网格上**跨 hex 一致**（相邻面三角剖分
    完全重合）。可用于 FVM 守恒 / 梯度收敛验证。

    返回 (vertices[N,3], cells[M,4])，所有单元正定向（带号体积 > 0）。
    """
    ny = nx if ny is None else ny
    nz = nx if nz is None else nz
    vx = np.linspace(0.0, 1.0, nx + 1)
    vy = np.linspace(0.0, 1.0, ny + 1)
    vz = np.linspace(0.0, 1.0, nz + 1)
    node = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(
        nx + 1, ny + 1, nz + 1)
    verts = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                verts.append([vx[i], vy[j], vz[k]])
    V = np.asarray(verts, float)
    # Kuhn 6-tet：绕主对角线 0–6 的环 {1,2,3,7,4,5}，逐段成 (0, a, b, 6)
    TRI6 = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n000 = node[i, j, k]
                n100 = node[i + 1, j, k]
                n110 = node[i + 1, j + 1, k]
                n010 = node[i, j + 1, k]
                n001 = node[i, j, k + 1]
                n101 = node[i + 1, j, k + 1]
                n111 = node[i + 1, j + 1, k + 1]
                n011 = node[i, j + 1, k + 1]
                local = [n000, n100, n110, n010, n001, n101, n111, n011]
                for tet in TRI6:
                    cells.append([local[t] for t in tet])
    C = np.asarray(cells, np.int64)
    signed = _tet_volume(V, C)
    if (signed <= 0.0).any():
        neg = signed < 0.0
        C[neg, [1, 3]] = C[neg, [3, 1]]
        signed = _tet_volume(V, C)
        if (signed <= 0.0).any():
            raise ValueError("cube_tet_mesh 存在非正体积单元，校验失败")
    return V, C
