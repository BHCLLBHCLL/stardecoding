# -*- coding: utf-8 -*-
"""STAR-CCM+ 图形窗口鼠标互动器。

默认键位（Simcenter STAR-CCM+ 文档《移动视图》）：
  左键拖动     绕按下时鼠标下的 3D 点旋转（不是原点 / 不是相机焦点）
  Ctrl+左键    滚动（绕垂直屏幕轴）
  中键拖动     缩放（向下放大、向上缩小），绕按下点
  滚轮         朝自己转 = 放大；缩放朝向光标下的点
  右键拖动     平移
  Ctrl+中键    平移

VTK 默认 vtkInteractorStyleTrackballCamera.Rotate 走 camera.Azimuth/Elevation，
绕的是 FocalPoint。ResetCamera 后焦点常在包围盒中心 ≈ 原点，模型会绕远点甩。
"""

import math

import numpy as np
import vtk


# 与 VTK MotionFactor=10 同量级（约 0.22°/像素 @ 900px）
_ROTATE_DEG_PER_PX = 0.20
_DOLLY_PX = 20.0
_DOLLY_STEP = 1.10
_WHEEL_STEP = 1.10
_CLICK_PX = 5.0

# vtkInteractorStyle 状态（避免个别绑定没有 VTKIS_* 属性）
_VTKIS_ROTATE = 1
_VTKIS_DOLLY = 4


def _norm(v):
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return v, 0.0
    return v / n, n


def _rodrigues(vec, axis, deg):
    axis, n = _norm(np.asarray(axis, dtype=float))
    if n < 1e-15:
        return vec
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    v = np.asarray(vec, dtype=float)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _display_to_world(ren, x, y, z):
    """屏幕点 → 世界坐标（必须除以齐次 w，否则会落到原点附近）。"""
    try:
        ren.SetDisplayPoint(float(x), float(y), float(z))
        ren.DisplayToWorld()
        wx, wy, wz, ww = ren.GetWorldPoint()
    except Exception:
        return None
    if abs(ww) < 1e-12:
        return None
    return (wx / ww, wy / ww, wz / ww)


def orbit_camera(cam, center, az_deg, el_deg):
    """绕固定世界点 center 转动相机（方位绕 ViewUp，仰俯绕右轴）。"""
    C = np.asarray(center, dtype=float).reshape(3)
    pos = np.array(cam.GetPosition(), dtype=float)
    fp = np.array(cam.GetFocalPoint(), dtype=float)
    up = np.array(cam.GetViewUp(), dtype=float)
    rel_p = _rodrigues(pos - C, up, az_deg)
    rel_f = _rodrigues(fp - C, up, az_deg)
    view, vn = _norm((C + rel_f) - (C + rel_p))
    if vn < 1e-15:
        return False
    right, rn = _norm(np.cross(view, up))
    if rn >= 1e-8:
        rel_p = _rodrigues(rel_p, right, el_deg)
        rel_f = _rodrigues(rel_f, right, el_deg)
        up = _rodrigues(up, right, el_deg)
    cam.SetPosition(*(C + rel_p))
    cam.SetFocalPoint(*(C + rel_f))
    cam.SetViewUp(*up)
    try:
        cam.OrthogonalizeViewUp()
    except Exception:
        pass
    return True


class StarCCMInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """左键绕按下时鼠标下的 3D 点旋转；中键缩放、右键平移。"""

    def __init__(self):
        super().__init__()
        self.SetMotionFactor(10.0)
        try:
            self.SetMouseWheelMotionFactor(0.5)
        except Exception:
            pass
        self._center = None
        self._lmb = None
        self._did_drag = False

    def _renderer(self):
        rwi = self.GetInteractor()
        if rwi is None:
            return None
        x, y = rwi.GetEventPosition()
        self.FindPokedRenderer(int(x), int(y))
        return self.GetCurrentRenderer()

    def _in_front_of_camera(self, cam, p):
        pos = np.array(cam.GetPosition(), dtype=float)
        fp = np.array(cam.GetFocalPoint(), dtype=float)
        view, n = _norm(fp - pos)
        if n < 1e-15:
            return True
        return float(np.dot(np.asarray(p, dtype=float) - pos, view)) > 1e-6

    def _world_on_plane(self, ren, x, y, plane_pt):
        """光标射线与过 plane_pt、垂直视线的平面求交。"""
        try:
            ren.SetWorldPoint(plane_pt[0], plane_pt[1], plane_pt[2], 1.0)
            ren.WorldToDisplay()
            z = ren.GetDisplayPoint()[2]
        except Exception:
            z = 0.5
        return _display_to_world(ren, x, y, z)

    def _world_under_cursor(self, x, y):
        """按下瞬间鼠标下的 3D 点：最近可见表面，而不是原点或远裁剪面。"""
        ren = self.GetCurrentRenderer()
        rwi = self.GetInteractor()
        if ren is None:
            return None
        cam = ren.GetActiveCamera()
        if cam is None:
            return None
        candidates = []

        zbuf = None
        try:
            zbuf = float(ren.GetZ(int(x), int(y)))
        except Exception:
            zbuf = None
        if zbuf is not None and 0.0005 < zbuf < 0.9995:
            p = _display_to_world(ren, x, y, zbuf)
            if p is not None and self._in_front_of_camera(cam, p):
                candidates.append(p)

        picker = None
        if rwi is not None:
            picker = rwi.GetPicker()
        if picker is None:
            picker = vtk.vtkCellPicker()
            try:
                picker.SetTolerance(0.005)
            except Exception:
                pass
        try:
            if picker.Pick(float(x), float(y), 0.0, ren):
                cell_ok = True
                try:
                    cell_ok = picker.GetCellId() >= 0 or picker.GetActor() is not None
                except Exception:
                    cell_ok = True
                if cell_ok:
                    p = tuple(picker.GetPickPosition())
                    if self._in_front_of_camera(cam, p):
                        candidates.append(p)
        except Exception:
            pass

        if candidates:
            pos = np.array(cam.GetPosition(), dtype=float)
            return tuple(min(candidates, key=lambda q: np.linalg.norm(np.array(q) - pos)))

        ref = self._center
        if ref is None:
            ref = tuple(cam.GetFocalPoint())
        p = self._world_on_plane(ren, x, y, ref)
        if p is not None and self._in_front_of_camera(cam, p):
            return p
        return tuple(ref)

    def _grab_center(self):
        rwi = self.GetInteractor()
        if rwi is None:
            self._center = None
            return
        x, y = rwi.GetEventPosition()
        self._center = self._world_under_cursor(x, y)

    def _clip_and_render(self):
        ren = self.GetCurrentRenderer()
        rwi = self.GetInteractor()
        if ren is not None:
            try:
                ren.ResetCameraClippingRange()
            except Exception:
                pass
        if rwi is not None:
            rwi.Render()

    def _zoom_by(self, factor):
        """factor > 1 放大（靠近中心 / 减小 ParallelScale）。"""
        if factor <= 0.0:
            return
        ren = self.GetCurrentRenderer()
        rwi = self.GetInteractor()
        if ren is None or rwi is None:
            return
        cam = ren.GetActiveCamera()
        if cam is None:
            return
        C = np.array(self._center if self._center is not None else cam.GetFocalPoint(),
                     dtype=float)
        if cam.GetParallelProjection():
            x, y = rwi.GetEventPosition()
            w0 = _display_to_world(ren, x, y, 0.5)
            try:
                ren.SetWorldPoint(C[0], C[1], C[2], 1.0)
                ren.WorldToDisplay()
                z = ren.GetDisplayPoint()[2]
            except Exception:
                z = 0.5
            w0 = _display_to_world(ren, x, y, z)
            cam.SetParallelScale(max(cam.GetParallelScale() / factor, 1e-9))
            w1 = _display_to_world(ren, x, y, z)
            if w0 is not None and w1 is not None:
                delta = np.array(w0, dtype=float) - np.array(w1, dtype=float)
                fp = np.array(cam.GetFocalPoint(), dtype=float) + delta
                pos = np.array(cam.GetPosition(), dtype=float) + delta
                cam.SetFocalPoint(*fp)
                cam.SetPosition(*pos)
        else:
            pos = np.array(cam.GetPosition(), dtype=float)
            fp = np.array(cam.GetFocalPoint(), dtype=float)
            cam.SetPosition(*(C + (pos - C) / factor))
            cam.SetFocalPoint(*(C + (fp - C) / factor))
        self._clip_and_render()

    def OnLeftButtonDown(self):
        rwi = self.GetInteractor()
        if rwi is None:
            return
        x, y = rwi.GetEventPosition()
        self.FindPokedRenderer(int(x), int(y))
        self._lmb = (int(x), int(y))
        self._did_drag = False
        self._grab_center()
        if rwi.GetControlKey():
            self.StartSpin()
        elif rwi.GetShiftKey():
            self.StartPan()
        else:
            self.StartRotate()

    def OnLeftButtonUp(self):
        rwi = self.GetInteractor()
        if rwi is not None and self._lmb is not None and not self._did_drag:
            x, y = rwi.GetEventPosition()
            if math.hypot(x - self._lmb[0], y - self._lmb[1]) <= _CLICK_PX:
                picker = rwi.GetPicker()
                ren = self.GetCurrentRenderer()
                if picker is not None and ren is not None:
                    try:
                        picker.Pick(float(x), float(y), 0.0, ren)
                    except Exception:
                        pass
        self._lmb = None
        super().OnLeftButtonUp()

    def OnMiddleButtonDown(self):
        rwi = self.GetInteractor()
        if rwi is None:
            return
        x, y = rwi.GetEventPosition()
        self.FindPokedRenderer(int(x), int(y))
        self._grab_center()
        if rwi.GetControlKey():
            self.StartPan()
        else:
            self.StartDolly()

    def OnMiddleButtonUp(self):
        super().OnMiddleButtonUp()
        self.EndDolly()
        self.EndPan()

    def OnRightButtonDown(self):
        rwi = self.GetInteractor()
        if rwi is None:
            return
        x, y = rwi.GetEventPosition()
        self.FindPokedRenderer(int(x), int(y))
        self.StartPan()

    def OnRightButtonUp(self):
        super().OnRightButtonUp()
        self.EndPan()
        self.EndDolly()

    def OnMouseMove(self):
        if self._lmb is not None:
            rwi = self.GetInteractor()
            if rwi is not None:
                x, y = rwi.GetEventPosition()
                if math.hypot(x - self._lmb[0], y - self._lmb[1]) > _CLICK_PX:
                    self._did_drag = True
        try:
            state = int(self.GetState())
        except Exception:
            state = -1
        # 自己处理旋转/缩放，避免 C++ Azimuth 仍绕 FocalPoint（原点）转一遍
        if state == _VTKIS_ROTATE:
            self.Rotate()
            return
        if state == _VTKIS_DOLLY:
            self.Dolly()
            return
        super().OnMouseMove()

    def Rotate(self):
        rwi = self.GetInteractor()
        ren = self.GetCurrentRenderer()
        if rwi is None or ren is None:
            return
        cam = ren.GetActiveCamera()
        if cam is None:
            return
        dx = rwi.GetEventPosition()[0] - rwi.GetLastEventPosition()[0]
        dy = rwi.GetEventPosition()[1] - rwi.GetLastEventPosition()[1]
        h = max(int(ren.GetSize()[1]), 1)
        speed = _ROTATE_DEG_PER_PX * (900.0 / float(h))
        C = self._center if self._center is not None else cam.GetFocalPoint()
        if not orbit_camera(cam, C, -dx * speed, dy * speed):
            return
        self._clip_and_render()

    def Dolly(self):
        rwi = self.GetInteractor()
        if rwi is None:
            return
        dy = rwi.GetEventPosition()[1] - rwi.GetLastEventPosition()[1]
        factor = math.pow(_DOLLY_STEP, -dy / _DOLLY_PX)
        self._zoom_by(factor)

    def OnMouseWheelForward(self):
        """滚轮远离自己 → 缩小。"""
        if self._renderer() is None:
            return
        self._grab_center()
        self._zoom_by(1.0 / _WHEEL_STEP)

    def OnMouseWheelBackward(self):
        """滚轮朝自己 → 放大。"""
        if self._renderer() is None:
            return
        self._grab_center()
        self._zoom_by(_WHEEL_STEP)


def install_starccm_interactor(iren):
    """把 STAR-CCM+ 互动器装到 vtkRenderWindowInteractor。"""
    style = StarCCMInteractorStyle()
    iren.SetInteractorStyle(style)
    try:
        iren.SetDesiredUpdateRate(30.0)
        iren.SetStillUpdateRate(0.0001)
    except Exception:
        pass
    return style
