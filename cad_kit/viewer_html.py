"""Self-contained, CSP-safe WebGL Interactive 3D CAD & Section-Cut Viewer Generator.

Works both:
  1. Inside Antigravity's chat / side-pane (`generative_ui` compliant: zero external JS CDNs,
     uses allowlisted Tailwind + semantic CSS theme tokens).
  2. Standalone in any desktop browser (`file:///.../viewer.html`).

Features:
  - Z-up turntable orbit (left-drag), pan (right-drag / shift-drag), zoom (wheel)
  - 8 one-click engineering camera presets: 前(-Y), 后(+Y), 左(-X), 右(+X), 顶(+Z), 底(-Z), 等轴俯, 等轴仰
  - Parallel Orthographic vs Perspective toggle + X-Ray mode + 10mm/50mm XY bed grid & XYZ axes
  - Real-time X / Y / Z section-cut sliders with amber cross-section interior highlighting
  - Floating 3D Feature Anchor pins ([H1], [S1]...) + clickable feature & parameter inspector
  - Optional Reference model & Volumetric Diff (+Added / -Removed) layer toggles
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


def _load_viewer_mesh(target: Any) -> trimesh.Trimesh:
    """Load mesh with lightweight viewer tessellation so HTML stays compact (~80-180 KB)."""
    if hasattr(target, "tessellate"):
        return shape_to_trimesh(target, angular_deflection=0.28, linear_deflection=0.15)
    if isinstance(target, (str, Path)):
        p = Path(target).expanduser().resolve()
        if p.suffix.lower() in (".step", ".stp", ".brep"):
            import build123d as bd
            part = bd.import_brep(str(p)) if p.suffix.lower() == ".brep" else bd.import_step(str(p))
            return shape_to_trimesh(part, angular_deflection=0.28, linear_deflection=0.15)
    return load_any_as_trimesh(target)


def _encode_f32(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return base64.b64encode(a.tobytes()).decode("ascii")


def _pack_mesh_layer(
    mesh: trimesh.Trimesh,
    name: str,
    label: str,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    visible: bool = True,
) -> dict[str, Any] | None:
    if mesh is None or len(mesh.faces) == 0:
        return None
    # Expand triangle Soup with crisp face/split normals via PyVista feature_angle=28
    poly = trimesh_to_pv(mesh).triangulate()
    faces = poly.faces.reshape(-1, 4)[:, 1:4]
    pts = np.asarray(poly.points, dtype=np.float32)
    tri_pos = pts[faces].reshape(-1, 3)

    if "Normals" in poly.point_data:
        nrms = np.asarray(poly.point_data["Normals"], dtype=np.float32)
        tri_nrm = nrms[faces].reshape(-1, 3)
    else:
        fn = np.asarray(mesh.face_normals, dtype=np.float32)
        tri_nrm = np.repeat(fn, 3, axis=0)

    # Extract sharp CAD feature edges as GL_LINES endpoint pairs
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

    return {
        "name": name,
        "label": label,
        "color": list(color),
        "opacity": float(opacity),
        "visible": bool(visible),
        "vertCount": int(len(tri_pos)),
        "posB64": _encode_f32(tri_pos),
        "nrmB64": _encode_f32(tri_nrm),
        "edgeVertCount": int(len(edge_pts)),
        "edgeB64": _encode_f32(edge_pts),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__ - 3D CAD 交互预览</title>
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
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-2">
      <!-- WebGL Viewport -->
      <div class="lg:col-span-3 bg-[var(--card)] border border-[var(--border)] rounded-lg relative overflow-hidden" style="height: 500px;" id="viewport-box">
        <canvas id="gl-canvas" class="w-full h-full block cursor-grab active:cursor-grabbing"></canvas>
        <div id="pins-layer" class="absolute inset-0 pointer-events-none overflow-hidden"></div>

        <!-- Floating Orientation & Controls Hint -->
        <div class="absolute left-2.5 bottom-2.5 px-2 py-1 rounded bg-[var(--background)]/85 border border-[var(--border)] text-xs text-[var(--muted-foreground)] pointer-events-none">
          左键旋转 · 右键/Shift平移 · 滚轮缩放 &nbsp;|&nbsp;
          <span style="color:#ef4444;font-weight:600;">X右(+)</span>
          <span style="color:#22c55e;font-weight:600;">Y后(+)</span>
          <span style="color:#3b82f6;font-weight:600;">Z上(+)</span>
          <span style="color:#f97316;font-weight:600;">■橙色=实心剖切面</span>
        </div>
      </div>

      <!-- Right Sidebar: Section Cut, Display & Feature Anchors -->
      <div class="bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 flex flex-col gap-3 text-xs" style="max-height: 500px; overflow-y: auto;">
        <!-- Display Toggles -->
        <div class="space-y-1.5 pb-2 border-b border-[var(--border)]">
          <div class="font-semibold text-[var(--foreground)]">视图与显示控制</div>
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
              <input type="checkbox" id="chk-pins" checked class="accent-[var(--primary)]" /> 特征ID标牌
            </label>
            <label class="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" id="chk-grid" checked class="accent-[var(--primary)]" /> 热床网格
            </label>
          </div>
          <div id="layers-list" class="flex flex-wrap gap-2 pt-1"></div>
        </div>

        <!-- Live X/Y/Z Section Cut Sliders -->
        <div class="space-y-1.5 pb-2 border-b border-[var(--border)]">
          <div class="flex items-center justify-between">
            <span class="font-semibold">实时剖面切割 (Section)</span>
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
          <div class="font-semibold">特征锚点 (点击聚焦)</div>
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

function b64ToF32(b64) {
  if (!b64) return new Float32Array(0);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// Matrix math helpers (column-major 4x4)
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

// Populate header & panels
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
varying vec3 vPos;
varying vec3 vNrm;
void main() {
  vPos = aPos;
  vNrm = aNrm;
  gl_Position = uMVP * vec4(aPos, 1.0);
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
varying vec3 vPos;
varying vec3 vCol;
void main() {
  vPos = aPos;
  vCol = aCol;
  gl_Position = uMVP * vec4(aPos, 1.0);
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

// Upload layers
const gpuLayers = CAD_DATA.layers.map((ly) => {
  const posArr = b64ToF32(ly.posB64);
  const nrmArr = b64ToF32(ly.nrmB64);
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

  return { ...ly, posBuf, nrmBuf, edgeBuf, edgeColBuf };
});

// Build bed grid (Z = minZ or 0) + RGB XYZ coordinate axes
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
      pts.push(-gridHalf, v, zGrid, gridHalf, v, zGrid);
      cols.push(c, c, c, c, c, c);
      pts.push(v, -gridHalf, zGrid, v, gridHalf, zGrid);
      cols.push(c, c, c, c, c, c);
    }
  }
  const axLen = gridHalf * 1.05;
  // X axis (Red: -X dashed-dim, +X bright)
  pts.push(-axLen, 0, zGrid, 0, 0, zGrid); cols.push(0.75, 0.35, 0.35, 0.75, 0.35, 0.35);
  pts.push(0, 0, zGrid, axLen, 0, zGrid);  cols.push(0.94, 0.26, 0.26, 0.94, 0.26, 0.26);
  // Y axis (Green: -Y Front, +Y Back)
  pts.push(0, -axLen, zGrid, 0, 0, zGrid); cols.push(0.25, 0.65, 0.35, 0.25, 0.65, 0.35);
  pts.push(0, 0, zGrid, 0, axLen, zGrid);  cols.push(0.13, 0.77, 0.37, 0.13, 0.77, 0.37);
  // Z axis (Blue: vertical at origin)
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

// Layer checkboxes UI
const layersListEl = document.getElementById('layers-list');
gpuLayers.forEach((ly, idx) => {
  const lbl = document.createElement('label');
  lbl.className = 'flex items-center gap-1 cursor-pointer px-1.5 py-0.5 rounded border border-[var(--border)]';
  const chk = document.createElement('input');
  chk.type = 'checkbox';
  chk.checked = ly.visible;
  chk.className = 'accent-[var(--primary)]';
  chk.addEventListener('change', () => { ly.visible = chk.checked; requestRender(); });
  const dot = document.createElement('span');
  dot.style.cssText = `display:inline-block;width:8px;height:8px;border-radius:99px;background:rgb(${Math.round(ly.color[0]*255)},${Math.round(ly.color[1]*255)},${Math.round(ly.color[2]*255)})`;
  lbl.appendChild(chk); lbl.appendChild(dot); lbl.appendChild(document.createTextNode(ly.label));
  layersListEl.appendChild(lbl);
});

// Camera state (Z-up spherical turntable: azim=0 looks from -Y Front! azim=+90 looks from +X Right!)
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
  clipFrac: [1.0, 1.0, 1.0],
};

// Pin DOM elements
const pinsLayer = document.getElementById('pins-layer');
const anchorsPanel = document.getElementById('anchors-panel');
const pinEls = (CAD_DATA.anchors || []).map((anc) => {
  const el = document.createElement('div');
  el.className = 'cad-pin px-1.5 py-0.5 rounded text-xs font-mono font-bold shadow border';
  el.style.cssText = 'background:#fef2f2;color:#991b1b;border-color:#f87171;font-size:11px;';
  el.textContent = anc.id;
  el.title = `${anc.id}: ${anc.desc} ${anc.spec ? '('+anc.spec+')' : ''} @ (${anc.pos.join(', ')}) mm`;
  el.addEventListener('click', () => {
    state.target = [...anc.pos];
    requestRender();
  });
  pinsLayer.appendChild(el);

  const row = document.createElement('div');
  row.className = 'p-1.5 rounded border border-[var(--border)] hover:bg-[var(--secondary)] cursor-pointer flex items-center justify-between gap-1';
  row.innerHTML = `<div><span class="font-mono font-bold text-[var(--primary)]">[${anc.id}]</span> <span>${anc.desc}</span></div><span class="font-mono text-[var(--muted-foreground)]">${anc.spec || ''}</span>`;
  row.addEventListener('click', () => { state.target = [...anc.pos]; requestRender(); });
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

// Camera presets matching unified convention: Front=-Y, Back=+Y, Left=-X, Right=+X, Top=+Z, Bottom=-Z
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
    if (p) {
      state.azimDeg = p.azim;
      state.elevDeg = p.elev;
      state.target = [...bcenter];
      requestRender();
    }
  });
});

// Toggles & section sliders
document.getElementById('chk-ortho').addEventListener('change', e => { state.ortho = e.target.checked; requestRender(); });
document.getElementById('chk-xray').addEventListener('change', e => { state.xray = e.target.checked; requestRender(); });
document.getElementById('chk-edges').addEventListener('change', e => { state.edges = e.target.checked; requestRender(); });
document.getElementById('chk-pins').addEventListener('change', e => { state.pins = e.target.checked; requestRender(); });
document.getElementById('chk-grid').addEventListener('change', e => { state.grid = e.target.checked; requestRender(); });

['x', 'y', 'z'].forEach((ax, idx) => {
  const rng = document.getElementById('rng-clip-' + ax);
  const valEl = document.getElementById('val-clip-' + ax);
  rng.addEventListener('input', () => {
    const f = Number(rng.value) / 1000;
    state.clipFrac[idx] = f;
    if (f >= 0.999) {
      valEl.textContent = '关闭';
    } else {
      const mm = bmin[idx] + f * (bmax[idx] - bmin[idx]);
      valEl.textContent = mm.toFixed(1) + ' mm';
    }
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

// Pointer interaction
let dragMode = null, lastX = 0, lastY = 0;
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('mousedown', e => {
  dragMode = (e.button === 2 || e.shiftKey) ? 'pan' : 'orbit';
  lastX = e.clientX; lastY = e.clientY;
});
window.addEventListener('mouseup', () => { dragMode = null; });
window.addEventListener('mousemove', e => {
  if (!dragMode) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
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
  const factor = Math.exp(-e.deltaY * 0.0012);
  state.zoom = Math.max(0.15, Math.min(12.0, state.zoom * factor));
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

let rafPending = false;
function requestRender() {
  if (!rafPending) {
    rafPending = true;
    requestAnimationFrame(renderFrame);
  }
}

function renderFrame() {
  rafPending = false;
  const box = document.getElementById('viewport-box');
  const dpr = window.devicePixelRatio || 1;
  const w = box.clientWidth, h = box.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
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
  // az=0, el=0 => camera at (0, -dist, 0) looking toward +Y (Front view!)
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

  const clipActive = state.clipFrac.map(f => f < 0.999 ? 1.0 : 0.0);
  const clipMax = state.clipFrac.map((f, i) => bmin[i] + f * (bmax[i] - bmin[i]));
  const anyClip = clipActive[0] + clipActive[1] + clipActive[2] > 0.5;

  // 1. Draw bed grid & axes
  if (state.grid) {
    gl.useProgram(progLine);
    gl.uniformMatrix4fv(gl.getUniformLocation(progLine, 'uMVP'), false, mvp);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uAlpha'), 0.65);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uUseClip'), 0.0);
    const aPos = gl.getAttribLocation(progLine, 'aPos');
    const aCol = gl.getAttribLocation(progLine, 'aCol');
    gl.bindBuffer(gl.ARRAY_BUFFER, gridGpu.pBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, gridGpu.cBuf);
    gl.enableVertexAttribArray(aCol);
    gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.LINES, 0, gridGpu.count);
  }

  // 2. Draw solid & translucent mesh layers (opaque first, translucent second)
  const ordered = [...gpuLayers].filter(l => l.visible).sort((a, b) => b.opacity - a.opacity);
  gl.useProgram(progMesh);
  gl.uniformMatrix4fv(gl.getUniformLocation(progMesh, 'uMVP'), false, mvp);
  gl.uniform3fv(gl.getUniformLocation(progMesh, 'uClipMax'), clipMax);
  gl.uniform3fv(gl.getUniformLocation(progMesh, 'uClipActive'), clipActive);

  ordered.forEach((ly) => {
    const effOpacity = state.xray ? Math.min(ly.opacity, 0.30) : ly.opacity;
    const transparent = effOpacity < 0.98;
    if (transparent) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
    } else {
      gl.disable(gl.BLEND);
      gl.depthMask(true);
    }
    gl.uniform3fv(gl.getUniformLocation(progMesh, 'uColor'), ly.color);
    gl.uniform1f(gl.getUniformLocation(progMesh, 'uOpacity'), effOpacity);
    gl.uniform1f(gl.getUniformLocation(progMesh, 'uShowCap'), (anyClip && !transparent) ? 1.0 : 0.0);

    const aPos = gl.getAttribLocation(progMesh, 'aPos');
    const aNrm = gl.getAttribLocation(progMesh, 'aNrm');
    gl.bindBuffer(gl.ARRAY_BUFFER, ly.posBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, ly.nrmBuf);
    gl.enableVertexAttribArray(aNrm);
    gl.vertexAttribPointer(aNrm, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, ly.vertCount);
  });
  gl.depthMask(true);
  gl.disable(gl.BLEND);

  // 3. Draw sharp CAD edges
  if (state.edges) {
    gl.useProgram(progLine);
    gl.uniformMatrix4fv(gl.getUniformLocation(progLine, 'uMVP'), false, mvp);
    gl.uniform3fv(gl.getUniformLocation(progLine, 'uClipMax'), clipMax);
    gl.uniform3fv(gl.getUniformLocation(progLine, 'uClipActive'), clipActive);
    gl.uniform1f(gl.getUniformLocation(progLine, 'uUseClip'), 1.0);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    const aPos = gl.getAttribLocation(progLine, 'aPos');
    const aCol = gl.getAttribLocation(progLine, 'aCol');
    ordered.forEach((ly) => {
      if (!ly.edgeVertCount) return;
      gl.uniform1f(gl.getUniformLocation(progLine, 'uAlpha'), ly.opacity < 0.5 ? 0.45 : 0.9);
      gl.bindBuffer(gl.ARRAY_BUFFER, ly.edgeBuf);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, ly.edgeColBuf);
      gl.enableVertexAttribArray(aCol);
      gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, ly.edgeVertCount);
    });
  }

  // 4. Update 2D projected Feature ID badges
  pinEls.forEach(({ anc, el }) => {
    if (!state.pins) { el.style.display = 'none'; return; }
    const scr = projectWorldToScreen(anc.pos, mvp, w, h);
    if (!scr || scr[0] < -20 || scr[0] > w + 20 || scr[1] < -20 || scr[1] > h + 20) {
      el.style.display = 'none';
    } else {
      el.style.display = 'block';
      el.style.left = scr[0].toFixed(1) + 'px';
      el.style.top = scr[1].toFixed(1) + 'px';
    }
  });
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
) -> Path:
    """Export a standalone interactive WebGL 3D CAD & Section-Cut viewer HTML file."""
    main_mesh = _load_viewer_mesh(target)
    bmin, bmax = main_mesh.bounds
    bsize = bmax - bmin
    bcenter = 0.5 * (bmin + bmax)

    layers: list[dict[str, Any]] = []
    main_layer = _pack_mesh_layer(main_mesh, "main", "当前零件", (0.23, 0.51, 0.96), opacity=1.0, visible=True)
    if main_layer:
        layers.append(main_layer)

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
