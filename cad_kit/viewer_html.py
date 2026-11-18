"""Self-contained, CSP-safe WebGL Interactive 3D CAD Viewer with Raycasting Picker & 3D Caliper.

Works both:
  1. Inside Antigravity's chat / side-pane (`generative_ui` compliant: zero external JS CDNs,
     uses allowlisted Tailwind + semantic CSS theme tokens).
  2. Standalone in any desktop browser (`file:///.../viewer.html`).

Features (v0.2):
  - Z-up turntable orbit (left-drag), pan (right-drag / shift-drag), zoom (wheel)
  - 8 one-click engineering camera presets: 前(-Y), 后(+Y), 左(-X), 右(+X), 顶(+Z), 底(-Z), 等轴俯, 等轴仰
  - Section-aware 3D Raycasting ("Click-to-Prompt"): click any face, sharp CAD edge, corner vertex,
    or hole anchor -> highlights 3D point + surface normal arrow & generates a copy-ready Agent prompt!
  - 3D Point-to-Point Digital Caliper (P1 ↔ P2): measures exact Euclidean distance + |ΔX|, |ΔY|, |ΔZ|
    components with 3D dimension lines and smart vertex / edge-midpoint / hole-center snapping!
  - Real-time X / Y / Z section-cut sliders with amber cross-section interior highlighting
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .inspector import shape_to_trimesh
from .renderer import extract_cad_edges, trimesh_to_pv
from .verify import load_any_as_trimesh


def _encode_bytes(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _encode_f32(arr: np.ndarray) -> str:
    return _encode_bytes(np.asarray(arr, dtype=np.float32))


def _load_viewer_mesh(target: Any) -> trimesh.Trimesh:
    """Load mesh with cached `viewer` LOD (<=35k tris via O(N) C++ QuadricClustering) so HTML never bloats!"""
    from .large_cad import load_mesh_cached

    return load_mesh_cached(target, tier="viewer")


def _pack_mesh_layer(
    mesh: trimesh.Trimesh,
    name: str,
    label: str,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    visible: bool = True,
    explode_vec: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any] | None:
    if mesh is None or len(mesh.faces) == 0:
        return None
    poly = trimesh_to_pv(mesh).triangulate()
    faces = np.asarray(poly.faces.reshape(-1, 4)[:, 1:4], dtype=np.uint32)
    pts = np.asarray(poly.points, dtype=np.float32)

    if "Normals" in poly.point_data and len(poly.point_data["Normals"]) == len(pts):
        nrms = np.asarray(poly.point_data["Normals"], dtype=np.float32)
    else:
        poly_n = poly.compute_normals(point_normals=True, cell_normals=False, split_vertices=True)
        pts = np.asarray(poly_n.points, dtype=np.float32)
        faces = np.asarray(poly_n.faces.reshape(-1, 4)[:, 1:4], dtype=np.uint32)
        nrms = np.asarray(poly_n.point_data["Normals"], dtype=np.float32)

    nrm_i8 = np.clip(np.round(nrms * 127.0), -127, 127).astype(np.int8)
    use_u16 = len(pts) < 65535
    idx_arr = faces.ravel().astype(np.uint16 if use_u16 else np.uint32)

    edges_pv = extract_cad_edges(poly, angle=24.0)
    edge_pts = np.zeros((0, 3), dtype=np.float32)
    if edges_pv.n_points > 0 and edges_pv.lines.size > 0:
        raw = edges_pv.lines
        e_verts = np.asarray(edges_pv.points, dtype=np.float32)
        segs: list[np.ndarray] = []
        idx = 0
        n_raw = len(raw)
        while idx < n_raw:
            cnt = int(raw[idx])
            ids = raw[idx + 1 : idx + 1 + cnt]
            idx += 1 + cnt
            for k in range(cnt - 1):
                segs.append(e_verts[ids[k]])
                segs.append(e_verts[ids[k + 1]])
        if segs:
            edge_pts = np.asarray(segs, dtype=np.float32)

    center_pt = [round(float(x), 3) for x in (0.5 * (mesh.bounds[0] + mesh.bounds[1]))]
    return {
        "name": name,
        "label": label,
        "color": list(color),
        "opacity": float(opacity),
        "visible": bool(visible),
        "explodeVec": [round(float(x), 3) for x in explode_vec],
        "center": center_pt,
        "vertCount": int(len(idx_arr)),
        "u16": bool(use_u16),
        "uPosB64": _encode_f32(pts),
        "uNrmI8B64": _encode_bytes(nrm_i8),
        "triIdxB64": _encode_bytes(idx_arr),
        "edgeVertCount": int(len(edge_pts)),
        "edgeB64": _encode_f32(edge_pts),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__ - 3D CAD 交互点选与卡尺预览</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    html, body { margin: 0; padding: 0; }
    .cad-pin {
      position: absolute;
      transform: translate(-50%, -100%);
      pointer-events: auto;
      cursor: pointer;
      user-select: none;
      transition: transform 0.08s ease;
    }
    .cad-pin:hover { transform: translate(-50%, -108%) scale(1.08); z-index: 30; }
    .meas-badge {
      position: absolute;
      transform: translate(-50%, -50%);
      pointer-events: none;
      user-select: none;
      z-index: 25;
    }
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-2">
  <div class="flex flex-col gap-2 max-w-full">
    <!-- Top Header & Camera Presets -->
    <div class="bg-[var(--card)] border border-[var(--border)] rounded-lg px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
      <div class="flex items-center gap-2 flex-wrap">
        <span class="font-semibold text-sm" id="hdr-title"></span>
        <span class="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[var(--secondary-foreground)] font-mono" id="hdr-mode"></span>
        <span class="font-mono text-[var(--muted-foreground)]" id="hdr-bbox"></span>
      </div>
      <div class="flex items-center gap-1 flex-wrap">
        <button data-cam="front" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">前(-Y)</button>
        <button data-cam="back" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">后(+Y)</button>
        <button data-cam="left" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">左(-X)</button>
        <button data-cam="right" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">右(+X)</button>
        <button data-cam="top" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">顶(+Z)</button>
        <button data-cam="bottom" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">底(-Z)</button>
        <button data-cam="iso_top" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)] font-medium">等轴俯</button>
        <button data-cam="iso_bot" class="cam-btn px-2 py-1 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">等轴仰</button>
      </div>
    </div>

    <!-- Main Viewport + Inspector Split -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-2">
      <!-- WebGL Viewport -->
      <div class="lg:col-span-8 bg-[var(--card)] border border-[var(--border)] rounded-lg relative overflow-hidden" style="height: 560px;" id="viewport-box">
        <canvas id="gl-canvas" class="w-full h-full block cursor-crosshair"></canvas>
        <div id="pins-layer" class="absolute inset-0 pointer-events-none overflow-hidden"></div>
        <div id="meas-overlay" class="absolute inset-0 pointer-events-none overflow-hidden"></div>

        <!-- Top-Left Tool Mode Switcher Overlay -->
        <div class="absolute left-2.5 top-2.5 flex flex-wrap items-center gap-1.5 bg-[var(--card)]/90 border border-[var(--border)] rounded-lg p-1 text-xs shadow">
          <button id="tool-pick" class="px-2.5 py-1 rounded font-medium bg-[var(--primary)] text-[var(--primary-foreground)]">
            🎯 点选面/边改件
          </button>
          <button id="tool-caliper" class="px-2.5 py-1 rounded font-medium hover:bg-[var(--secondary)] text-[var(--foreground)]">
            📏 3D 两点卡尺
          </button>
          <label class="flex items-center gap-1 px-2 py-0.5 border-l border-[var(--border)] cursor-pointer text-[var(--muted-foreground)]" title="靠近孔心锚点、锐边端点或边中点时自动精准吸附">
            <input type="checkbox" id="chk-snap" checked class="accent-[var(--primary)]" /> 吸附顶点/孔心
          </label>
          <button id="btn-clear-pick" class="px-1.5 py-0.5 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)]" title="清除点选与卡尺标记">清除</button>
        </div>

        <!-- Live Hover Coordinate Readout -->
        <div id="hover-readout" class="absolute right-2.5 top-2.5 px-2 py-1 rounded bg-[var(--card)]/90 border border-[var(--border)] text-xs font-mono text-[var(--muted-foreground)] pointer-events-none">
          单击模型任意表面/边拾取坐标 · 拖拽旋转
        </div>

        <!-- Bottom Legend -->
        <div class="absolute left-2.5 bottom-2.5 px-2 py-1 rounded bg-[var(--background)]/85 border border-[var(--border)] text-xs text-[var(--muted-foreground)] pointer-events-none">
          单击拾取/测距 · 左键拖拽旋转 · 右键/Shift平移 · 滚轮缩放 &nbsp;|&nbsp;
          <span style="color:#ef4444;font-weight:600;">X右(+)</span>
          <span style="color:#22c55e;font-weight:600;">Y后(+)</span>
          <span style="color:#3b82f6;font-weight:600;">Z上(+)</span>
          <span style="color:#f97316;font-weight:600;">■橙色=剖切面(支持剖面内壁测距!)</span>
        </div>
      </div>

      <!-- Right Sidebar: Click-to-Prompt, 3D Caliper, Section Cut & Anchors -->
      <div class="lg:col-span-4 bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 flex flex-col gap-3 text-xs" style="max-height: 560px; overflow-y: auto;">

        <!-- Panel A: Click-to-Prompt / 3D Caliper Readout -->
        <div id="panel-pick" class="space-y-2 pb-2.5 border-b border-[var(--border)]">
          <div class="flex items-center justify-between">
            <span class="font-semibold text-[var(--foreground)]">🎯 指哪打哪 (点选生成修改口令)</span>
            <span id="pick-snap-badge" class="font-mono text-[var(--muted-foreground)]">未点选</span>
          </div>
          <div id="pick-detail" class="p-2 rounded bg-[var(--background)] border border-[var(--border)] font-mono text-[var(--muted-foreground)] leading-relaxed">
            在左侧模型上<b>单击任意面、台阶、孔心或边线</b>，自动提取精确三维坐标与表面法向。
          </div>
          <div class="flex flex-wrap gap-1">
            <button data-tpl="hole" class="tpl-btn px-2 py-0.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">+ 在此打孔</button>
            <button data-tpl="offset" class="tpl-btn px-2 py-0.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">↔ 此面偏移/加厚</button>
            <button data-tpl="fillet" class="tpl-btn px-2 py-0.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">╭ 附近边倒圆角</button>
            <button data-tpl="slot" class="tpl-btn px-2 py-0.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">▭ 在此开槽</button>
          </div>
          <textarea id="prompt-box" rows="3" class="w-full rounded p-1.5 bg-[var(--background)] border border-[var(--border)] text-[var(--foreground)] font-mono text-xs" placeholder="点选模型位置后将自动生成零歧义修改指令，可直接复制发给 Agent..."></textarea>
          <button id="btn-copy-prompt" class="w-full py-1.5 rounded font-semibold bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90">
            📋 一键复制修改口令发给 Agent
          </button>
        </div>

        <!-- Panel B: 3D Caliper Readout (shown in Caliper mode) -->
        <div id="panel-caliper" class="hidden space-y-2 pb-2.5 border-b border-[var(--border)]">
          <div class="flex items-center justify-between">
            <span class="font-semibold text-[var(--foreground)]">📏 3D 数字卡尺 (两点精确测距)</span>
            <button id="btn-reset-caliper" class="px-1.5 py-0.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">重新选点</button>
          </div>
          <div id="caliper-detail" class="p-2 rounded bg-[var(--background)] border border-[var(--border)] font-mono space-y-1 leading-relaxed">
            <div class="text-[var(--muted-foreground)]">请在模型上依次单击 <b>起点 P1</b> 与 <b>终点 P2</b>（可先拖动剖切滑块测量内部壁厚或沉孔台阶深度！）</div>
          </div>
          <button id="btn-copy-caliper" class="w-full py-1.5 rounded font-semibold bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90">
            📋 复制实测尺寸与修改指令
          </button>
        </div>

        <!-- Panel Explode: Multi-Part Assembly Explosion Slider (auto-shown when explodeVec exists) -->
        <div id="panel-explode" class="hidden space-y-1.5 pb-2.5 border-b border-[var(--border)]">
          <div class="flex items-center justify-between">
            <span class="font-semibold text-[var(--foreground)]">💥 装配体动态爆炸拆解</span>
            <span id="val-explode" class="font-mono font-bold text-[var(--primary)]">0% (闭合)</span>
          </div>
          <input type="range" id="rng-explode" min="0" max="100" value="0" class="w-full accent-[var(--primary)]" />
        </div>

        <!-- Display Toggles -->
        <div class="space-y-1.5 pb-2 border-b border-[var(--border)]">
          <div class="font-semibold text-[var(--foreground)]">视图与图层开关</div>
          <div class="flex flex-wrap gap-2 pt-0.5">
            <label class="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" id="chk-ortho" checked class="accent-[var(--primary)]" /> 正交无畸变
            </label>
            <label class="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" id="chk-xray" class="accent-[var(--primary)]" /> X-Ray透视
            </label>
            <label class="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" id="chk-edges" checked class="accent-[var(--primary)]" /> CAD边线
            </label>
            <label class="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" id="chk-pins" checked class="accent-[var(--primary)]" /> 特征ID红牌
            </label>
            <label class="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" id="chk-grid" checked class="accent-[var(--primary)]" /> 热床网格
            </label>
          </div>
          <div id="layers-list" class="flex flex-wrap gap-1.5 pt-1"></div>
        </div>

        <!-- Live X/Y/Z Section Cut Sliders -->
        <div class="space-y-1.5 pb-2 border-b border-[var(--border)]">
          <div class="flex items-center justify-between">
            <span class="font-semibold">实时实心剖切 (支持剖切后测内腔!)</span>
            <button id="btn-reset-clip" class="px-1.5 py-0.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)]">重置剖切</button>
          </div>
          <div class="space-y-1">
            <div class="flex items-center justify-between font-mono">
              <span>X 剖切 (左→右)</span><span id="val-clip-x">关闭</span>
            </div>
            <input type="range" id="rng-clip-x" min="0" max="1000" value="1000" class="w-full accent-[var(--primary)]" />
          </div>
          <div class="space-y-1">
            <div class="flex items-center justify-between font-mono">
              <span>Y 剖切 (前→后)</span><span id="val-clip-y">关闭</span>
            </div>
            <input type="range" id="rng-clip-y" min="0" max="1000" value="1000" class="w-full accent-[var(--primary)]" />
          </div>
          <div class="space-y-1">
            <div class="flex items-center justify-between font-mono">
              <span>Z 剖切 (下→上)</span><span id="val-clip-z">关闭</span>
            </div>
            <input type="range" id="rng-clip-z" min="0" max="1000" value="1000" class="w-full accent-[var(--primary)]" />
          </div>
        </div>

        <!-- Feature Anchors List -->
        <div class="space-y-1.5 pb-2 border-b border-[var(--border)]">
          <div class="font-semibold">已注册特征锚点 (点击直接选中/测距)</div>
          <div id="anchors-panel" class="space-y-1"></div>
        </div>

        <!-- Key Parameters -->
        <div class="space-y-1">
          <div class="font-semibold">当前建模参数 (Params)</div>
          <div id="params-panel" class="space-y-0.5 font-mono text-[var(--muted-foreground)]"></div>
        </div>
      </div>
    </div>
  </div>

<script>
const CAD_DATA = __CAD_JSON__;

function b64ToBytes(b64) {
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
function b64ToF32(b64) {
  const b = b64ToBytes(b64);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
}
function unpackIndexedLayer(ly) {
  if (ly.posB64) return { posArr: b64ToF32(ly.posB64), nrmArr: b64ToF32(ly.nrmB64) };
  const uPos = b64ToF32(ly.uPosB64);
  const nrmBytes = b64ToBytes(ly.uNrmI8B64);
  const uNrm = new Int8Array(nrmBytes.buffer, nrmBytes.byteOffset, nrmBytes.byteLength);
  const idxBytes = b64ToBytes(ly.triIdxB64);
  const idxArr = ly.u16
    ? new Uint16Array(idxBytes.buffer, idxBytes.byteOffset, idxBytes.byteLength / 2)
    : new Uint32Array(idxBytes.buffer, idxBytes.byteOffset, idxBytes.byteLength / 4);
  const n = idxArr.length;
  const posArr = new Float32Array(n * 3);
  const nrmArr = new Float32Array(n * 3);
  const inv127 = 1.0 / 127.0;
  for (let i = 0; i < n; i++) {
    const vi = idxArr[i] * 3, oi = i * 3;
    posArr[oi]   = uPos[vi];     posArr[oi+1] = uPos[vi+1];   posArr[oi+2] = uPos[vi+2];
    nrmArr[oi]   = uNrm[vi]*inv127; nrmArr[oi+1] = uNrm[vi+1]*inv127; nrmArr[oi+2] = uNrm[vi+2]*inv127;
  }
  return { posArr, nrmArr };
}

// 4x4 Column-Major Matrix Helpers + Exact Matrix Inverse for Unproject Raycasting
function mat4Identity() {
  const m = new Float32Array(16);
  m[0]=1; m[5]=1; m[10]=1; m[15]=1;
  return m;
}
function mat4Mul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
    }
  }
  return out;
}
function mat4Ortho(l, r, b, t, near, far) {
  const m = new Float32Array(16);
  m[0] = 2/(r-l); m[5] = 2/(t-b); m[10] = -2/(far-near);
  m[12] = -(r+l)/(r-l); m[13] = -(t+b)/(t-b); m[14] = -(far+near)/(far-near); m[15] = 1;
  return m;
}
function mat4Persp(fovRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovRad / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f; m[10] = (far + near) / (near - far);
  m[11] = -1; m[14] = (2 * far * near) / (near - far);
  return m;
}
function mat4LookAt(eye, target, up) {
  let zx = eye[0]-target[0], zy = eye[1]-target[1], zz = eye[2]-target[2];
  let zl = Math.hypot(zx, zy, zz) || 1; zx/=zl; zy/=zl; zz/=zl;
  let xx = up[1]*zz - up[2]*zy, xy = up[2]*zx - up[0]*zz, xz = up[0]*zy - up[1]*zx;
  let xl = Math.hypot(xx, xy, xz);
  if (xl < 1e-5) { xx = 1; xy = 0; xz = 0; xl = 1; }
  xx/=xl; xy/=xl; xz/=xl;
  let yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;
  const m = mat4Identity();
  m[0]=xx; m[4]=xy; m[8]=xz;  m[12]=-(xx*eye[0]+xy*eye[1]+xz*eye[2]);
  m[1]=yx; m[5]=yy; m[9]=yz;  m[13]=-(yx*eye[0]+yy*eye[1]+yz*eye[2]);
  m[2]=zx; m[6]=zy; m[10]=zz; m[14]=-(zx*eye[0]+zy*eye[1]+zz*eye[2]);
  return m;
}
function mat4Invert(m) {
  const inv = new Float32Array(16);
  inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
  inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
  inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
  inv[12]= -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
  inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
  inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
  inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
  inv[13]= m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
  inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
  inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
  inv[10]= m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
  inv[14]= -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
  inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
  inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
  inv[11]= -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
  inv[15]= m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
  let det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
  if (Math.abs(det) < 1e-12) return null;
  det = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}
function unprojectNDC(ndcX, ndcY, ndcZ, invMVP) {
  const x = invMVP[0]*ndcX + invMVP[4]*ndcY + invMVP[8]*ndcZ + invMVP[12];
  const y = invMVP[1]*ndcX + invMVP[5]*ndcY + invMVP[9]*ndcZ + invMVP[13];
  const z = invMVP[2]*ndcX + invMVP[6]*ndcY + invMVP[10]*ndcZ + invMVP[14];
  const w = invMVP[3]*ndcX + invMVP[7]*ndcY + invMVP[11]*ndcZ + invMVP[15] || 1.0;
  return [x/w, y/w, z/w];
}
function projectWorldToScreen(pt, mvp, w, h) {
  const x = pt[0], y = pt[1], z = pt[2];
  const cx = mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12];
  const cy = mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13];
  const cz = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14];
  const cw = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
  if (cw <= 1e-5) return null;
  const ndcX = cx / cw, ndcY = cy / cw, ndcZ = cz / cw;
  if (ndcZ < -1.2 || ndcZ > 1.2) return null;
  return [(ndcX * 0.5 + 0.5) * w, (0.5 - ndcY * 0.5) * h];
}
function classifyNormalSemantic(n) {
  if (!n) return '点/边特征';
  const axes = [
    [[0, -1, 0], '正面 Front (-Y)'],
    [[0,  1, 0], '背面 Back (+Y)'],
    [[-1, 0, 0], '左侧面 Left (-X)'],
    [[ 1, 0, 0], '右侧面 Right (+X)'],
    [[0,  0, 1], '顶面 Top (+Z)'],
    [[0, 0, -1], '底面 Bottom (-Z)'],
  ];
  for (const [ref, name] of axes) {
    if (n[0]*ref[0] + n[1]*ref[1] + n[2]*ref[2] > 0.96) return name;
  }
  return `斜/曲面法向 (${n.map(v=>v.toFixed(2)).join(', ')})`;
}

// Header init
document.getElementById('hdr-title').textContent = CAD_DATA.title;
document.getElementById('hdr-mode').textContent = 'MODE: ' + (CAD_DATA.mode || 'fdm').toUpperCase();
const bs = CAD_DATA.bbox.size_mm;
document.getElementById('hdr-bbox').textContent = `尺寸 ${bs[0]} × ${bs[1]} × ${bs[2]} mm`;

const canvas = document.getElementById('gl-canvas');
const gl = canvas.getContext('webgl', { antialias: true, alpha: false, stencil: false });

const VS_MESH = `
attribute vec3 aPos;
attribute vec3 aNrm;
uniform mat4 uMVP;
uniform vec3 uOffset;
varying vec3 vPos;
varying vec3 vNrm;
void main() {
  vec3 wp = aPos + uOffset;
  vPos = wp;
  vNrm = aNrm;
  gl_Position = uMVP * vec4(wp, 1.0);
}`;

const FS_MESH = `
precision highp float;
varying vec3 vPos;
varying vec3 vNrm;
uniform vec3 uColor;
uniform float uOpacity;
uniform vec3 uClipMax;
uniform vec3 uClipActive;
uniform float uShowCap;
void main() {
  if (uClipActive.x > 0.5 && vPos.x > uClipMax.x) discard;
  if (uClipActive.y > 0.5 && vPos.y > uClipMax.y) discard;
  if (uClipActive.z > 0.5 && vPos.z > uClipMax.z) discard;

  if (!gl_FrontFacing && uShowCap > 0.5) {
    gl_FragColor = vec4(0.97, 0.45, 0.09, uOpacity);
    return;
  }
  vec3 N = normalize(gl_FrontFacing ? vNrm : -vNrm);
  vec3 L1 = normalize(vec3(0.55, -0.75, 0.85));
  vec3 L2 = normalize(vec3(-0.65, 0.60, -0.35));
  float d1 = max(dot(N, L1), 0.0);
  float d2 = max(dot(N, L2), 0.0);
  vec3 lit = uColor * (0.36 + 0.52 * d1 + 0.22 * d2);
  gl_FragColor = vec4(lit, uOpacity);
}`;

const VS_LINE = `
attribute vec3 aPos;
attribute vec3 aCol;
uniform mat4 uMVP;
uniform vec3 uOffset;
varying vec3 vPos;
varying vec3 vCol;
void main() {
  vec3 wp = aPos + uOffset;
  vPos = wp;
  vCol = aCol;
  gl_Position = uMVP * vec4(wp, 1.0);
}`;

const FS_LINE = `
precision highp float;
varying vec3 vPos;
varying vec3 vCol;
uniform float uAlpha;
uniform vec3 uClipMax;
uniform vec3 uClipActive;
uniform float uUseClip;
void main() {
  if (uUseClip > 0.5) {
    if (uClipActive.x > 0.5 && vPos.x > uClipMax.x) discard;
    if (uClipActive.y > 0.5 && vPos.y > uClipMax.y) discard;
    if (uClipActive.z > 0.5 && vPos.z > uClipMax.z) discard;
  }
  gl_FragColor = vec4(vCol, uAlpha);
}`;

function compileProg(vsSrc, fsSrc) {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, vsSrc); gl.compileShader(vs);
  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, fsSrc); gl.compileShader(fs);
  const p = gl.createProgram();
  gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
  return p;
}

const progMesh = compileProg(VS_MESH, FS_MESH);
const progLine = compileProg(VS_LINE, FS_LINE);

// Keep CPU Float32 arrays alongside GPU buffers for exact raycasting & vertex/edge snapping!
const gpuLayers = CAD_DATA.layers.map((ly) => {
  const { posArr, nrmArr } = unpackIndexedLayer(ly);
  const edgeArr = b64ToF32(ly.edgeB64);

  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.STATIC_DRAW);

  const nrmBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
  gl.bufferData(gl.ARRAY_BUFFER, nrmArr, gl.STATIC_DRAW);

  const edgeBuf = gl.createBuffer();
  const edgeCols = new Float32Array(edgeArr.length);
  for (let i = 0; i < edgeCols.length; i += 3) { edgeCols[i]=0.06; edgeCols[i+1]=0.09; edgeCols[i+2]=0.16; }
  gl.bindBuffer(gl.ARRAY_BUFFER, edgeBuf);
  gl.bufferData(gl.ARRAY_BUFFER, edgeArr, gl.STATIC_DRAW);

  const edgeColBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, edgeColBuf);
  gl.bufferData(gl.ARRAY_BUFFER, edgeCols, gl.STATIC_DRAW);

  return { ...ly, posArr, nrmArr, edgeArr, posBuf, nrmBuf, edgeBuf, edgeColBuf };
});

// Dynamic overlay line buffer (picked normal arrow, crosshair, 3D caliper line & ΔX/ΔY/ΔZ legs)
const dynPosBuf = gl.createBuffer();
const dynColBuf = gl.createBuffer();

const bmin = CAD_DATA.bbox.min_mm;
const bmax = CAD_DATA.bbox.max_mm;
const bcenter = CAD_DATA.bbox.center_mm;
const diag = Math.max(Math.hypot(bs[0], bs[1], bs[2]), 10.0);

function buildGridBuffers() {
  const gridHalf = Math.ceil(Math.max(Math.abs(bmin[0]), Math.abs(bmax[0]), Math.abs(bmin[1]), Math.abs(bmax[1]), diag*0.65) / 10) * 10;
  const step = gridHalf <= 80 ? 10 : 20;
  const zGrid = Math.abs(bmin[2]) < 0.2 ? 0.0 : bmin[2];
  const pts = [], cols = [];
  for (let v = -gridHalf; v <= gridHalf; v += step) {
    const isMajor = (v % (step*5) === 0);
    const c = isMajor ? 0.45 : 0.62;
    if (v !== 0) {
      pts.push(-gridHalf, v, zGrid, gridHalf, v, zGrid); cols.push(c, c, c, c, c, c);
      pts.push(v, -gridHalf, zGrid, v, gridHalf, zGrid); cols.push(c, c, c, c, c, c);
    }
  }
  const axLen = gridHalf * 1.05;
  pts.push(-axLen, 0, zGrid, 0, 0, zGrid); cols.push(0.75, 0.35, 0.35, 0.75, 0.35, 0.35);
  pts.push(0, 0, zGrid, axLen, 0, zGrid);  cols.push(0.94, 0.26, 0.26, 0.94, 0.26, 0.26);
  pts.push(0, -axLen, zGrid, 0, 0, zGrid); cols.push(0.25, 0.65, 0.35, 0.25, 0.65, 0.35);
  pts.push(0, 0, zGrid, 0, axLen, zGrid);  cols.push(0.13, 0.77, 0.37, 0.13, 0.77, 0.37);
  pts.push(0, 0, zGrid, 0, 0, zGrid + axLen*0.7); cols.push(0.23, 0.51, 0.96, 0.23, 0.51, 0.96);

  const pBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STATIC_DRAW);
  const cBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(cols), gl.STATIC_DRAW);
  return { pBuf, cBuf, count: pts.length / 3 };
}
const gridGpu = buildGridBuffers();

// State
const state = {
  target: [...bcenter],
  azimDeg: 38,
  elevDeg: 30,
  zoom: 1.0,
  ortho: true,
  xray: false,
  edges: true,
  pins: true,
  grid: true,
  snap: true,
  tool: 'pick',       // 'pick' | 'caliper'
  explode: 0.0,       // 0.0 ~ 1.0 assembly explosion factor
  clipFrac: [1.0, 1.0, 1.0],
  picked: null,       // { pt:[x,y,z], normal:[nx,ny,nz], faceLabel, snapLabel, layerLabel }
  caliperPts: [],     // [ pt1Obj, pt2Obj ]
  lastMVP: mat4Identity(),
  lastW: 800,
  lastH: 560,
};

// Layer checkboxes
const layersListEl = document.getElementById('layers-list');
gpuLayers.forEach((ly) => {
  const lbl = document.createElement('label');
  lbl.className = 'flex items-center gap-1 cursor-pointer px-1.5 py-0.5 rounded border border-[var(--border)]';
  const chk = document.createElement('input');
  chk.type = 'checkbox'; chk.checked = ly.visible; chk.className = 'accent-[var(--primary)]';
  chk.addEventListener('change', () => { ly.visible = chk.checked; requestRender(); });
  const dot = document.createElement('span');
  dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:99px;background:rgb(${Math.round(ly.color[0]*255)},${Math.round(ly.color[1]*255)},${Math.round(ly.color[2]*255)})`;
  lbl.appendChild(chk); lbl.appendChild(dot); lbl.appendChild(document.createTextNode(ly.label));
  layersListEl.appendChild(lbl);
});

// Tool mode switcher
const btnToolPick = document.getElementById('tool-pick');
const btnToolCal = document.getElementById('tool-caliper');
const panelPick = document.getElementById('panel-pick');
const panelCal = document.getElementById('panel-caliper');

function setToolMode(m) {
  state.tool = m;
  const activeCls = 'px-2.5 py-1 rounded font-medium bg-[var(--primary)] text-[var(--primary-foreground)]';
  const idleCls = 'px-2.5 py-1 rounded font-medium hover:bg-[var(--secondary)] text-[var(--foreground)]';
  btnToolPick.className = m === 'pick' ? activeCls : idleCls;
  btnToolCal.className = m === 'caliper' ? activeCls : idleCls;
  panelPick.classList.toggle('hidden', m !== 'pick');
  panelCal.classList.toggle('hidden', m !== 'caliper');
  requestRender();
}
btnToolPick.addEventListener('click', () => setToolMode('pick'));
btnToolCal.addEventListener('click', () => setToolMode('caliper'));
document.getElementById('chk-snap').addEventListener('change', e => { state.snap = e.target.checked; });
document.getElementById('btn-clear-pick').addEventListener('click', () => {
  state.picked = null; state.caliperPts = [];
  updatePickUI(); updateCaliperUI(); requestRender();
});
document.getElementById('btn-reset-caliper').addEventListener('click', () => {
  state.caliperPts = []; updateCaliperUI(); requestRender();
});

// Section-aware CPU Raycasting + Smart Anchor/Vertex/Edge-Midpoint Snapping
function isPointClipped(pt) {
  for (let i = 0; i < 3; i++) {
    if (state.clipFrac[i] < 0.999) {
      const mx = bmin[i] + state.clipFrac[i] * (bmax[i] - bmin[i]);
      if (pt[i] > mx + 1e-4) return true;
    }
  }
  return false;
}

function raycastScene(sx, sy) {
  const w = state.lastW, h = state.lastH;
  // 1. Check screen-space smart snap to Feature Anchors first (highest semantic value!)
  if (state.snap) {
    let bestAnc = null, bestDist = 11.0;
    for (const anc of (CAD_DATA.anchors || [])) {
      if (isPointClipped(anc.pos)) continue;
      const scr = projectWorldToScreen(anc.pos, state.lastMVP, w, h);
      if (!scr) continue;
      const d = Math.hypot(scr[0] - sx, scr[1] - sy);
      if (d < bestDist) { bestDist = d; bestAnc = anc; }
    }
    if (bestAnc) {
      const nrm = bestAnc.normal || [0, 0, 1];
      return {
        pt: [...bestAnc.pos],
        normal: [...nrm],
        faceLabel: classifyNormalSemantic(nrm),
        snapLabel: `吸附锚点 [${bestAnc.id}] ${bestAnc.desc}`,
        anchorId: bestAnc.id,
        layerLabel: '特征锚点',
      };
    }
  }

  // 2. Möller–Trumbore ray-triangle intersection across visible layers
  const invMVP = mat4Invert(state.lastMVP);
  if (!invMVP) return null;
  const ndcX = (sx / w) * 2.0 - 1.0;
  const ndcY = 1.0 - (sy / h) * 2.0;
  const pNear = unprojectNDC(ndcX, ndcY, -1.0, invMVP);
  const pFar  = unprojectNDC(ndcX, ndcY,  1.0, invMVP);
  let dx = pFar[0] - pNear[0], dy = pFar[1] - pNear[1], dz = pFar[2] - pNear[2];
  const dl = Math.hypot(dx, dy, dz) || 1.0;
  dx /= dl; dy /= dl; dz /= dl;

  let minT = Infinity, hitPt = null, hitNrm = null, hitLayer = null;

  for (const ly of gpuLayers) {
    if (!ly.visible) continue;
    const ev = ly.explodeVec || [0, 0, 0];
    const ox = ev[0] * state.explode, oy = ev[1] * state.explode, oz = ev[2] * state.explode;
    const P = ly.posArr;
    for (let i = 0; i < P.length; i += 9) {
      const ax=P[i]+ox,   ay=P[i+1]+oy, az=P[i+2]+oz;
      const bx=P[i+3]+ox, by=P[i+4]+oy, bz=P[i+5]+oz;
      const cx=P[i+6]+ox, cy=P[i+7]+oy, cz=P[i+8]+oz;
      const e1x=bx-ax, e1y=by-ay, e1z=bz-az;
      const e2x=cx-ax, e2y=cy-ay, e2z=cz-az;
      const hx = dy*e2z - dz*e2y, hy = dz*e2x - dx*e2z, hz = dx*e2y - dy*e2x;
      const det = e1x*hx + e1y*hy + e1z*hz;
      if (Math.abs(det) < 1e-8) continue;
      const invDet = 1.0 / det;
      const tx = pNear[0]-ax, ty = pNear[1]-ay, tz = pNear[2]-az;
      const u = (tx*hx + ty*hy + tz*hz) * invDet;
      if (u < 0.0 || u > 1.0) continue;
      const qx = ty*e1z - tz*e1y, qy = tz*e1x - tx*e1z, qz = tx*e1y - ty*e1x;
      const v = (dx*qx + dy*qy + dz*qz) * invDet;
      if (v < 0.0 || u + v > 1.0) continue;
      const t = (e2x*qx + e2y*qy + e2z*qz) * invDet;
      if (t <= 0.01 || t >= minT) continue;

      const px = pNear[0] + t*dx, py = pNear[1] + t*dy, pz = pNear[2] + t*dz;
      if (isPointClipped([px, py, pz])) continue;

      minT = t;
      hitPt = [px, py, pz];
      // Geometric face normal oriented toward camera ray
      let nx = e1y*e2z - e1z*e2y, ny = e1z*e2x - e1x*e2z, nz = e1x*e2y - e1y*e2x;
      const nl = Math.hypot(nx, ny, nz) || 1.0;
      nx/=nl; ny/=nl; nz/=nl;
      if (nx*dx + ny*dy + nz*dz > 0) { nx=-nx; ny=-ny; nz=-nz; }
      hitNrm = [nx, ny, nz];
      hitLayer = ly.label;
    }
  }

  if (!hitPt) return null;

  // 3. Optional snap to nearby sharp CAD edge vertex or edge midpoint (within 9px on screen & close in depth)
  let snapLabel = '表面自由点';
  if (state.snap) {
    let bestSnapPt = null, bestPxDist = 9.5, bestKind = '';
    const snapTolWorld = Math.max(diag * 0.08, 4.0);
    for (const ly of gpuLayers) {
      if (!ly.visible || !ly.edgeArr) continue;
      const ev = ly.explodeVec || [0, 0, 0];
      const ox = ev[0] * state.explode, oy = ev[1] * state.explode, oz = ev[2] * state.explode;
      const E = ly.edgeArr;
      for (let i = 0; i < E.length; i += 6) {
        const v1 = [E[i]+ox, E[i+1]+oy, E[i+2]+oz];
        const v2 = [E[i+3]+ox, E[i+4]+oy, E[i+5]+oz];
        const mid = [0.5*(v1[0]+v2[0]), 0.5*(v1[1]+v2[1]), 0.5*(v1[2]+v2[2])];
        for (const [cand, kindName] of [[v1, '锐边顶点'], [v2, '锐边顶点'], [mid, '锐边中点']]) {
          if (isPointClipped(cand)) continue;
          if (Math.hypot(cand[0]-hitPt[0], cand[1]-hitPt[1], cand[2]-hitPt[2]) > snapTolWorld) continue;
          const scr = projectWorldToScreen(cand, state.lastMVP, w, h);
          if (!scr) continue;
          const d = Math.hypot(scr[0]-sx, scr[1]-sy);
          if (d < bestPxDist) { bestPxDist = d; bestSnapPt = cand; bestKind = kindName; }
        }
      }
    }
    if (bestSnapPt) {
      hitPt = bestSnapPt;
      snapLabel = `已吸附: ${bestKind}`;
    }
  }

  hitPt = hitPt.map(v => Math.round(v * 100) / 100);
  return {
    pt: hitPt,
    normal: hitNrm,
    faceLabel: classifyNormalSemantic(hitNrm),
    snapLabel,
    layerLabel: hitLayer,
  };
}

// UI updaters for Click-to-Prompt & 3D Caliper
const pickSnapBadge = document.getElementById('pick-snap-badge');
const pickDetailEl = document.getElementById('pick-detail');
const promptBoxEl = document.getElementById('prompt-box');
const caliperDetailEl = document.getElementById('caliper-detail');

function buildPromptText(hit, tplKind) {
  if (!hit) return '';
  const coordStr = `(${hit.pt[0].toFixed(2)}, ${hit.pt[1].toFixed(2)}, ${hit.pt[2].toFixed(2)}) mm`;
  const ancPrefix = hit.anchorId ? `特征 [${hit.anchorId}] ` : '';
  const base = `修改零件 ${CAD_DATA.title}：针对 ${ancPrefix}位置 ${coordStr} [所在面朝向: ${hit.faceLabel}]，`;
  if (tplKind === 'hole')   return base + `在此处垂直表面打一个 M3 沉头通孔（深度打穿）。`;
  if (tplKind === 'offset') return base + `将该表面沿外法向偏移 +2.0 mm（加厚/外扩）。`;
  if (tplKind === 'fillet') return base + `将该位置附近的边缘倒圆角 R2.0 mm。`;
  if (tplKind === 'slot')   return base + `在此处开一个宽 8.0 mm × 长 14.0 mm 的通槽。`;
  return base + `请执行修改：`;
}

function updatePickUI() {
  const h = state.picked;
  if (!h) {
    pickSnapBadge.textContent = '未点选';
    pickDetailEl.innerHTML = '在左侧模型上<b>单击任意面、台阶、孔心或边线</b>，自动提取精确三维坐标与表面法向。';
    promptBoxEl.value = '';
    return;
  }
  pickSnapBadge.textContent = h.snapLabel;
  pickDetailEl.innerHTML = `
    <div><b>拾取坐标</b>: <span class="text-[var(--primary)] font-bold">(${h.pt[0].toFixed(2)}, ${h.pt[1].toFixed(2)}, ${h.pt[2].toFixed(2)}) mm</span></div>
    <div><b>表面朝向</b>: ${h.faceLabel}</div>
    <div><b>所属图层</b>: ${h.layerLabel}</div>
  `;
  promptBoxEl.value = buildPromptText(h, null);
}

document.querySelectorAll('.tpl-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (!state.picked) return;
    promptBoxEl.value = buildPromptText(state.picked, btn.dataset.tpl);
    promptBoxEl.focus();
  });
});

function copyWithFeedback(btnEl, text) {
  if (!text) return;
  const orig = btnEl.textContent;
  navigator.clipboard.writeText(text).then(() => {
    btnEl.textContent = '✓ 已复制到剪贴板！直接粘贴发给 Agent 即可';
    setTimeout(() => { btnEl.textContent = orig; }, 1800);
  }).catch(() => {
    promptBoxEl.select();
    document.execCommand('copy');
    btnEl.textContent = '✓ 已复制！';
    setTimeout(() => { btnEl.textContent = orig; }, 1800);
  });
}

document.getElementById('btn-copy-prompt').addEventListener('click', function() {
  copyWithFeedback(this, promptBoxEl.value);
});

let lastCaliperPrompt = '';
function updateCaliperUI() {
  const pts = state.caliperPts;
  const overlay = document.getElementById('meas-overlay');
  overlay.innerHTML = '';
  lastCaliperPrompt = '';

  if (pts.length === 0) {
    caliperDetailEl.innerHTML = '<div class="text-[var(--muted-foreground)]">请在模型上依次单击 <b>起点 P1</b> 与 <b>终点 P2</b>（支持吸附顶点与孔心！）</div>';
    return;
  }
  if (pts.length === 1) {
    const p1 = pts[0].pt;
    caliperDetailEl.innerHTML = `
      <div><span style="color:#f59e0b;font-weight:700;">● 起点 P1</span>: (${p1[0].toFixed(2)}, ${p1[1].toFixed(2)}, ${p1[2].toFixed(2)}) mm <span class="text-[var(--muted-foreground)]">[${pts[0].snapLabel}]</span></div>
      <div class="text-[var(--primary)] font-semibold">👉 请继续在模型上单击终点 P2...</div>
    `;
    return;
  }
  const p1 = pts[0].pt, p2 = pts[1].pt;
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1], dz = p2[2] - p1[2];
  const dist = Math.hypot(dx, dy, dz);
  lastCaliperPrompt = `零件 ${CAD_DATA.title} 实测两点间距：P1(${p1.map(v=>v.toFixed(2)).join(', ')}) ↔ P2(${p2.map(v=>v.toFixed(2)).join(', ')}) 当前总距 = ${dist.toFixed(2)} mm (|ΔX|=${Math.abs(dx).toFixed(2)}, |ΔY|=${Math.abs(dy).toFixed(2)}, |ΔZ|=${Math.abs(dz).toFixed(2)} mm)。请将该间距调整为：`;
  caliperDetailEl.innerHTML = `
    <div><span style="color:#f59e0b;font-weight:700;">● P1</span>: (${p1[0].toFixed(2)}, ${p1[1].toFixed(2)}, ${p1[2].toFixed(2)})</div>
    <div><span style="color:#ec4899;font-weight:700;">● P2</span>: (${p2[0].toFixed(2)}, ${p2[1].toFixed(2)}, ${p2[2].toFixed(2)})</div>
    <div class="pt-1 border-t border-[var(--border)] text-sm font-bold text-[var(--primary)]">直线总距 L = ${dist.toFixed(2)} mm</div>
    <div class="grid grid-cols-3 gap-1 pt-0.5">
      <span style="color:#ef4444;">|ΔX| ${Math.abs(dx).toFixed(2)}</span>
      <span style="color:#22c55e;">|ΔY| ${Math.abs(dy).toFixed(2)}</span>
      <span style="color:#3b82f6;">|ΔZ| ${Math.abs(dz).toFixed(2)}</span>
    </div>
  `;
}

document.getElementById('btn-copy-caliper').addEventListener('click', function() {
  copyWithFeedback(this, lastCaliperPrompt);
});

// Handle a confirmed pick hit (from 3D canvas click OR sidebar anchor click)
function handleHitSelection(hit) {
  if (!hit) return;
  if (state.tool === 'pick') {
    state.picked = hit;
    updatePickUI();
  } else {
    if (state.caliperPts.length >= 2) state.caliperPts = [hit];
    else state.caliperPts.push(hit);
    updateCaliperUI();
  }
  requestRender();
}

// Pin DOM elements & Sidebar Anchor List
const pinsLayer = document.getElementById('pins-layer');
const anchorsPanel = document.getElementById('anchors-panel');
const pinEls = (CAD_DATA.anchors || []).map((anc) => {
  const el = document.createElement('div');
  el.className = 'cad-pin px-1.5 py-0.5 rounded text-xs font-mono font-bold shadow border';
  el.style.cssText = 'background:#fef2f2;color:#991b1b;border-color:#f87171;font-size:11px;';
  el.textContent = anc.id;
  el.title = `${anc.id}: ${anc.desc} ${anc.spec ? '('+anc.spec+')' : ''} @ (${anc.pos.join(', ')}) mm`;
  const selectAnchor = (ev) => {
    ev.stopPropagation();
    const nrm = anc.normal || [0, 0, 1];
    handleHitSelection({
      pt: [...anc.pos],
      normal: [...nrm],
      faceLabel: classifyNormalSemantic(nrm),
      snapLabel: `特征锚点 [${anc.id}]`,
      anchorId: anc.id,
      layerLabel: '特征锚点',
    });
  };
  el.addEventListener('click', selectAnchor);
  pinsLayer.appendChild(el);

  const row = document.createElement('div');
  row.className = 'p-1.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)] cursor-pointer flex items-center justify-between gap-1';
  row.innerHTML = `<div><span class="font-mono font-bold text-[var(--primary)]">[${anc.id}]</span> <span>${anc.desc}</span></div><span class="font-mono text-[var(--muted-foreground)]">${anc.spec || ''}</span>`;
  row.addEventListener('click', selectAnchor);
  anchorsPanel.appendChild(row);
  return { anc, el };
});
if (!pinEls.length) {
  anchorsPanel.innerHTML = '<div class="text-[var(--muted-foreground)]">暂无注册特征锚点</div>';
}

// Params list
const paramsPanel = document.getElementById('params-panel');
const pEntries = Object.entries(CAD_DATA.params || {});
if (pEntries.length) {
  pEntries.forEach(([k, v]) => {
    const d = document.createElement('div');
    d.className = 'flex justify-between gap-2';
    d.innerHTML = `<span>${k}</span><span class="text-[var(--foreground)]">${v}</span>`;
    paramsPanel.appendChild(d);
  });
} else {
  paramsPanel.textContent = '-';
}

// Camera presets
const CAM_PRESETS = {
  front:   { azim: 0,    elev: 0 },
  back:    { azim: 180,  elev: 0 },
  right:   { azim: 90,   elev: 0 },
  left:    { azim: -90,  elev: 0 },
  top:     { azim: 0,    elev: 89.9 },
  bottom:  { azim: 0,    elev: -89.9 },
  iso_top: { azim: 38,   elev: 30 },
  iso_bot: { azim: -142, elev: -30 },
};
document.querySelectorAll('.cam-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const p = CAM_PRESETS[btn.dataset.cam];
    if (p) { state.azimDeg = p.azim; state.elevDeg = p.elev; state.target = [...bcenter]; requestRender(); }
  });
});

document.getElementById('chk-ortho').addEventListener('change', e => { state.ortho = e.target.checked; requestRender(); });
document.getElementById('chk-xray').addEventListener('change', e => { state.xray = e.target.checked; requestRender(); });
document.getElementById('chk-edges').addEventListener('change', e => { state.edges = e.target.checked; requestRender(); });
document.getElementById('chk-pins').addEventListener('change', e => { state.pins = e.target.checked; requestRender(); });
document.getElementById('chk-grid').addEventListener('change', e => { state.grid = e.target.checked; requestRender(); });

const hasExplode = gpuLayers.some(l => l.explodeVec && Math.hypot(...l.explodeVec) > 0.1);
if (hasExplode) {
  document.getElementById('panel-explode').classList.remove('hidden');
  const rngExp = document.getElementById('rng-explode');
  const valExp = document.getElementById('val-explode');
  rngExp.addEventListener('input', () => {
    state.explode = Number(rngExp.value) / 100.0;
    valExp.textContent = state.explode === 0 ? '0% (闭合)' : `${rngExp.value}% (拆解)`;
    requestRender();
  });
}

['x', 'y', 'z'].forEach((ax, idx) => {
  const rng = document.getElementById('rng-clip-' + ax);
  const valEl = document.getElementById('val-clip-' + ax);
  rng.addEventListener('input', () => {
    const f = Number(rng.value) / 1000;
    state.clipFrac[idx] = f;
    valEl.textContent = f >= 0.999 ? '关闭' : (bmin[idx] + f * (bmax[idx] - bmin[idx])).toFixed(1) + ' mm';
    requestRender();
  });
});
document.getElementById('btn-reset-clip').addEventListener('click', () => {
  state.clipFrac = [1, 1, 1];
  ['x', 'y', 'z'].forEach(ax => {
    document.getElementById('rng-clip-' + ax).value = 1000;
    document.getElementById('val-clip-' + ax).textContent = '关闭';
  });
  requestRender();
});

// Pointer interaction: distinguish Click (< 4px move) vs Orbit/Pan Drag!
let dragMode = null, downX = 0, downY = 0, lastX = 0, lastY = 0, movedDist = 0;
const hoverReadout = document.getElementById('hover-readout');

canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('mousedown', e => {
  dragMode = (e.button === 2 || e.shiftKey) ? 'pan' : 'orbit';
  downX = lastX = e.clientX;
  downY = lastY = e.clientY;
  movedDist = 0;
});
window.addEventListener('mouseup', e => {
  if (dragMode && movedDist < 4.5 && e.button === 0 && !e.shiftKey && e.target === canvas) {
    const rect = canvas.getBoundingClientRect();
    const hit = raycastScene(e.clientX - rect.left, e.clientY - rect.top);
    if (hit) handleHitSelection(hit);
  }
  dragMode = null;
});
window.addEventListener('mousemove', e => {
  if (!dragMode) {
    if (e.target === canvas) {
      const rect = canvas.getBoundingClientRect();
      const hit = raycastScene(e.clientX - rect.left, e.clientY - rect.top);
      hoverReadout.textContent = hit
        ? `(${hit.pt[0].toFixed(2)}, ${hit.pt[1].toFixed(2)}, ${hit.pt[2].toFixed(2)}) mm · ${hit.snapLabel}`
        : '单击模型任意表面/边拾取坐标 · 拖拽旋转';
    }
    return;
  }
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  movedDist += Math.hypot(dx, dy);
  if (dragMode === 'orbit') {
    state.azimDeg -= dx * 0.45;
    state.elevDeg = Math.max(-89.9, Math.min(89.9, state.elevDeg + dy * 0.45));
  } else {
    const scale = (diag / state.zoom) * 0.0016;
    const az = state.azimDeg * Math.PI / 180;
    const rx = Math.cos(az), ry = Math.sin(az);
    state.target[0] -= dx * rx * scale;
    state.target[1] -= dx * ry * scale;
    state.target[2] += dy * scale;
  }
  requestRender();
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  state.zoom = Math.max(0.15, Math.min(12.0, state.zoom * Math.exp(-e.deltaY * 0.0012)));
  requestRender();
}, { passive: false });

function getBgClearColor() {
  const isLight = document.documentElement.classList.contains('light');
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--card').trim();
  if (bg.startsWith('#') && bg.length === 7) {
    return [parseInt(bg.slice(1,3),16)/255, parseInt(bg.slice(3,5),16)/255, parseInt(bg.slice(5,7),16)/255];
  }
  return isLight ? [0.97, 0.98, 0.99] : [0.09, 0.11, 0.15];
}

// Build 3D overlay line primitives for picked crosshair/normal & 3D caliper dimension lines
function buildOverlayLines() {
  const pts = [], cols = [];
  const crossR = Math.max(diag * 0.035, 1.2);
  function addCross(p, col) {
    pts.push(p[0]-crossR, p[1], p[2], p[0]+crossR, p[1], p[2]); cols.push(...col, ...col);
    pts.push(p[0], p[1]-crossR, p[2], p[0], p[1]+crossR, p[2]); cols.push(...col, ...col);
    pts.push(p[0], p[1], p[2]-crossR, p[0], p[1], p[2]+crossR); cols.push(...col, ...col);
  }
  if (state.tool === 'pick' && state.picked) {
    const p = state.picked.pt, n = state.picked.normal || [0,0,1];
    addCross(p, [0.02, 0.78, 0.95]);
    const nLen = Math.max(diag * 0.14, 5.0);
    pts.push(p[0], p[1], p[2], p[0]+n[0]*nLen, p[1]+n[1]*nLen, p[2]+n[2]*nLen);
    cols.push(0.02, 0.78, 0.95, 0.99, 0.80, 0.08);
  }
  if (state.caliperPts.length >= 1) {
    addCross(state.caliperPts[0].pt, [0.96, 0.62, 0.04]);
  }
  if (state.caliperPts.length === 2) {
    const p1 = state.caliperPts[0].pt, p2 = state.caliperPts[1].pt;
    addCross(p2, [0.93, 0.28, 0.60]);
    // Direct Euclidean measurement line (bright yellow-magenta)
    pts.push(p1[0],p1[1],p1[2], p2[0],p2[1],p2[2]);
    cols.push(0.96, 0.62, 0.04, 0.93, 0.28, 0.60);
    // Orthogonal ΔX / ΔY / ΔZ component guide legs
    const px = [p2[0], p1[1], p1[2]], pxy = [p2[0], p2[1], p1[2]];
    pts.push(...p1, ...px);  cols.push(0.94,0.26,0.26, 0.94,0.26,0.26);
    pts.push(...px, ...pxy); cols.push(0.13,0.77,0.37, 0.13,0.77,0.37);
    pts.push(...pxy, ...p2); cols.push(0.23,0.51,0.96, 0.23,0.51,0.96);
  }
  // Draw explosion guide lines connecting home center -> exploded center
  if (state.explode > 0.01) {
    gpuLayers.forEach((ly) => {
      if (!ly.visible || !ly.center || !ly.explodeVec) return;
      const ev = ly.explodeVec;
      if (Math.hypot(ev[0], ev[1], ev[2]) < 0.2) return;
      const c0 = ly.center;
      const c1 = [c0[0] + ev[0]*state.explode, c0[1] + ev[1]*state.explode, c0[2] + ev[2]*state.explode];
      pts.push(...c0, ...c1);
      cols.push(...ly.color, ...ly.color);
    });
  }
  return { pts: new Float32Array(pts), cols: new Float32Array(cols), count: pts.length / 3 };
}

let rafPending = false;
function requestRender() {
  if (!rafPending) { rafPending = true; requestAnimationFrame(renderFrame); }
}

function renderFrame() {
  rafPending = false;
  const box = document.getElementById('viewport-box');
  const dpr = window.devicePixelRatio || 1;
  const w = box.clientWidth, h = box.clientHeight;
  state.lastW = w; state.lastH = h;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  gl.viewport(0, 0, canvas.width, canvas.height);
  const bg = getBgClearColor();
  gl.clearColor(bg[0], bg[1], bg[2], 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  const aspect = Math.max(w / Math.max(h, 1), 0.2);
  const az = state.azimDeg * Math.PI / 180;
  const el = state.elevDeg * Math.PI / 180;
  const dist = diag * 2.4 / state.zoom;
  const cx = state.target[0] + dist * Math.cos(el) * Math.sin(az);
  const cy = state.target[1] - dist * Math.cos(el) * Math.cos(az);
  const cz = state.target[2] + dist * Math.sin(el);
  const upVec = Math.abs(state.elevDeg) > 88 ? [0, state.elevDeg > 0 ? 1 : -1, 0] : [0, 0, 1];

  const viewMat = mat4LookAt([cx, cy, cz], state.target, upVec);
  const halfH = (diag * 0.62) / state.zoom;
  const projMat = state.ortho
    ? mat4Ortho(-halfH * aspect, halfH * aspect, -halfH, halfH, 0.1, diag * 20)
    : mat4Persp(38 * Math.PI / 180, aspect, Math.max(0.2, dist * 0.05), diag * 20);
  const mvp = mat4Mul(projMat, viewMat);
  state.lastMVP = mvp;

  const clipActive = state.clipFrac.map(f => f < 0.999 ? 1.0 : 0.0);
  const clipMax = state.clipFrac.map((f, i) => bmin[i] + f * (bmax[i] - bmin[i]));
  const anyClip = clipActive[0] + clipActive[1] + clipActive[2] > 0.5;

  // 1. Grid
  if (state.grid) {
    gl.useProgram(progLine);
    gl.uniformMatrix4fv(gl.getUniformLocation(progLine, 'uMVP'), false, mvp);
    gl.uniform3fv(gl.getUniformLocation(progLine, 'uOffset'), [0, 0, 0]);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uAlpha'), 0.65);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uUseClip'), 0.0);
    const aPos = gl.getAttribLocation(progLine, 'aPos'), aCol = gl.getAttribLocation(progLine, 'aCol');
    gl.bindBuffer(gl.ARRAY_BUFFER, gridGpu.pBuf); gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, gridGpu.cBuf); gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, gridGpu.count);
  }

  // 2. Meshes
  const ordered = [...gpuLayers].filter(l => l.visible).sort((a, b) => b.opacity - a.opacity);
  gl.useProgram(progMesh);
  gl.uniformMatrix4fv(gl.getUniformLocation(progMesh, 'uMVP'), false, mvp);
  gl.uniform3fv(gl.getUniformLocation(progMesh, 'uClipMax'), clipMax);
  gl.uniform3fv(gl.getUniformLocation(progMesh, 'uClipActive'), clipActive);

  ordered.forEach((ly) => {
    const ev = ly.explodeVec || [0, 0, 0];
    gl.uniform3fv(gl.getUniformLocation(progMesh, 'uOffset'), [ev[0]*state.explode, ev[1]*state.explode, ev[2]*state.explode]);
    const effOpacity = state.xray ? Math.min(ly.opacity, 0.30) : ly.opacity;
    const transparent = effOpacity < 0.98;
    if (transparent) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
    else { gl.disable(gl.BLEND); gl.depthMask(true); }
    gl.uniform3fv(gl.getUniformLocation(progMesh, 'uColor'), ly.color);
    gl.uniform1f(gl.getUniformLocation(progMesh, 'uOpacity'), effOpacity);
    gl.uniform1f(gl.getUniformLocation(progMesh, 'uShowCap'), (anyClip && !transparent) ? 1.0 : 0.0);

    const aPos = gl.getAttribLocation(progMesh, 'aPos'), aNrm = gl.getAttribLocation(progMesh, 'aNrm');
    gl.bindBuffer(gl.ARRAY_BUFFER, ly.posBuf); gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, ly.nrmBuf); gl.enableVertexAttribArray(aNrm); gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, ly.vertCount);
  });
  gl.depthMask(true); gl.disable(gl.BLEND);

  // 3. CAD Sharp Edges
  if (state.edges) {
    gl.useProgram(progLine);
    gl.uniformMatrix4fv(gl.getUniformLocation(progLine, 'uMVP'), false, mvp);
    gl.uniform3fv(gl.getUniformLocation(progLine, 'uClipMax'), clipMax);
    gl.uniform3fv(gl.getUniformLocation(progLine, 'uClipActive'), clipActive);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uUseClip'), 1.0);
    const aPos = gl.getAttribLocation(progLine, 'aPos'), aCol = gl.getAttribLocation(progLine, 'aCol');
    ordered.forEach((ly) => {
      if (!ly.edgeVertCount) return;
      const ev = ly.explodeVec || [0, 0, 0];
      gl.uniform3fv(gl.getUniformLocation(progLine, 'uOffset'), [ev[0]*state.explode, ev[1]*state.explode, ev[2]*state.explode]);
      gl.uniform1f(gl.getUniformLocation(progLine, 'uAlpha'), ly.opacity < 0.5 ? 0.45 : 0.9);
      gl.bindBuffer(gl.ARRAY_BUFFER, ly.edgeBuf); gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, ly.edgeColBuf); gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, ly.edgeVertCount);
    });
  }

  // 4. Picked Point Normal Arrow & 3D Caliper Overlay Lines (always visible on top!)
  const ov = buildOverlayLines();
  if (ov.count > 0) {
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(progLine);
    gl.uniformMatrix4fv(gl.getUniformLocation(progLine, 'uMVP'), false, mvp);
    gl.uniform3fv(gl.getUniformLocation(progLine, 'uOffset'), [0, 0, 0]);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uAlpha'), 1.0);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uUseClip'), 0.0);
    const aPos = gl.getAttribLocation(progLine, 'aPos'), aCol = gl.getAttribLocation(progLine, 'aCol');
    gl.bindBuffer(gl.ARRAY_BUFFER, dynPosBuf); gl.bufferData(gl.ARRAY_BUFFER, ov.pts, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, dynColBuf); gl.bufferData(gl.ARRAY_BUFFER, ov.cols, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, ov.count);
    gl.enable(gl.DEPTH_TEST);
  }

  // 5. Update 2D projected Feature ID badges & Caliper floating readout badge
  pinEls.forEach(({ anc, el }) => {
    if (!state.pins || isPointClipped(anc.pos)) { el.style.display = 'none'; return; }
    const scr = projectWorldToScreen(anc.pos, mvp, w, h);
    if (!scr || scr[0] < -20 || scr[0] > w + 20 || scr[1] < -20 || scr[1] > h + 20) {
      el.style.display = 'none';
    } else {
      el.style.display = 'block';
      el.style.left = scr[0].toFixed(1) + 'px';
      el.style.top = scr[1].toFixed(1) + 'px';
    }
  });

  const measOverlay = document.getElementById('meas-overlay');
  measOverlay.innerHTML = '';
  if (state.caliperPts.length === 2) {
    const p1 = state.caliperPts[0].pt, p2 = state.caliperPts[1].pt;
    const mid = [0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1]), 0.5*(p1[2]+p2[2])];
    const scr = projectWorldToScreen(mid, mvp, w, h);
    if (scr) {
      const dist = Math.hypot(p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]);
      const b = document.createElement('div');
      b.className = 'meas-badge px-2 py-0.5 rounded font-mono font-bold text-xs shadow-md';
      b.style.cssText = 'background:#1e1b4b;color:#fde047;border:1px solid #f59e0b;';
      b.textContent = `L = ${dist.toFixed(2)} mm`;
      b.style.left = scr[0].toFixed(1) + 'px';
      b.style.top = scr[1].toFixed(1) + 'px';
      measOverlay.appendChild(b);
    }
  }
}

window.addEventListener('resize', requestRender);
window.addEventListener('themechange', requestRender);
requestRender();
</script>
</body>
</html>
"""


def export_viewer_html(
    target: Any,
    out_html: str | Path,
    title: str = "Part",
    mode: str = "fdm",
    params: dict[str, Any] | None = None,
    anchors: list[dict[str, Any]] | None = None,
    ref_files: list[str | Path] | None = None,
    diff_meshes: dict[str, trimesh.Trimesh] | None = None,
    assembly_items: list[dict[str, Any]] | None = None,
) -> Path:
    """Export a standalone interactive WebGL 3D CAD Picker, Caliper, Explode & Section-Cut viewer HTML file."""
    layers: list[dict[str, Any]] = []
    if assembly_items:
        all_meshes: list[trimesh.Trimesh] = []
        for it in assembly_items:
            m_it = _load_viewer_mesh(it["shape"])
            all_meshes.append(m_it)
            ly = _pack_mesh_layer(
                m_it,
                name=str(it["id"]),
                label=str(it["label"]),
                color=tuple(it["rgb"]),
                opacity=float(it.get("opacity", 1.0)),
                visible=True,
                explode_vec=tuple(it.get("explode_vec", (0.0, 0.0, 0.0))),
            )
            if ly:
                layers.append(ly)
        combined = trimesh.util.concatenate(all_meshes)
        bmin, bmax = combined.bounds
    else:
        main_mesh = _load_viewer_mesh(target)
        bmin, bmax = main_mesh.bounds
        main_layer = _pack_mesh_layer(main_mesh, "main", "当前零件", (0.23, 0.51, 0.96), opacity=1.0, visible=True)
        if main_layer:
            layers.append(main_layer)

    bsize = bmax - bmin
    bcenter = 0.5 * (bmin + bmax)

    if ref_files:
        ref_list = [_load_viewer_mesh(rf) for rf in ref_files if Path(rf).exists()]
        if ref_list:
            ref_layer = _pack_mesh_layer(
                trimesh.util.concatenate(ref_list),
                "ref",
                "参考原件",
                (0.58, 0.64, 0.72),
                opacity=0.32,
                visible=True,
            )
            if ref_layer:
                layers.append(ref_layer)

    if diff_meshes:
        add_m = diff_meshes.get("added_mesh")
        rem_m = diff_meshes.get("removed_mesh")
        if add_m is not None and len(add_m.faces) > 0:
            ly_add = _pack_mesh_layer(add_m, "diff_add", "+新增结构", (0.06, 0.73, 0.51), opacity=0.95, visible=False)
            if ly_add:
                layers.append(ly_add)
        if rem_m is not None and len(rem_m.faces) > 0:
            ly_rem = _pack_mesh_layer(rem_m, "diff_rem", "-切除部分", (0.94, 0.27, 0.27), opacity=0.50, visible=False)
            if ly_rem:
                layers.append(ly_rem)

    payload = {
        "title": title,
        "mode": mode,
        "bbox": {
            "min_mm": [round(float(x), 3) for x in bmin],
            "max_mm": [round(float(x), 3) for x in bmax],
            "size_mm": [round(float(x), 2) for x in bsize],
            "center_mm": [round(float(x), 3) for x in bcenter],
        },
        "params": params or {},
        "anchors": anchors or [],
        "layers": layers,
    }

    html_str = (
        HTML_TEMPLATE.replace("__TITLE__", str(title))
        .replace("__CAD_JSON__", json.dumps(payload, ensure_ascii=False))
    )
    out_path = Path(out_html).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")
    return out_path
