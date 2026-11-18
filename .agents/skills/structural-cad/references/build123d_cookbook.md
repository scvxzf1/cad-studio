# `build123d` 结构件建模与现有模型修改实战范式 (Cookbook)

遵循本手册中的建模顺序与拓扑选择范式，可避免 95% 以上的 OpenCASCADE 内核报错（如 `StdFail_NotDone`、共面布尔残留薄皮、索引漂移）。

---

## 一、 黄金建模五步顺序（严禁把圆角放在最前面乱导！）

在 `part.py` 的 `build()` 函数中，严格推荐按以下顺序组织几何运算：

1. **主体毛坯相加 (`+`)**：先拼合主底板、立板、安装凸台（`make_mounting_boss`）、三角加强筋（`make_rib`）。注意两块相加实体之间最好有 **`0.1~0.5mm` 嵌入重叠量**，避免零厚度擦边接触导致非流形边。
2. **大开槽与减重腔切除 (`-`)**：切掉主内腔、避空槽、走线开口。切割体在敞口侧务必留有 **`+1~2mm` 越界余量（overcut）**，绝不与表面刚好共面相减。
3. **结构大圆角 (`safe_fillet`)**：对立板转角、外轮廓竖直边（`part.edges().filter_by(Axis.Z)`）导圆角。
4. **精密孔位与螺母腔切除 (`-`)**：使用 `make_counterbore_cutter`、`make_heatset_cutter`、`make_nut_trap_cutter`、`make_slot_cutter` 打孔，并同步调用 `reg.mark(...)`。
5. **底面防象脚微倒角 (`safe_chamfer`)**：最后对底面轮廓（`part.faces().sort_by(Axis.Z)[0].outer_wire().edges()`）执行 `tol.bed_chamfer`。

---

## 二、 稳健的拓扑选择器（禁止硬编码下标 `faces()[7]`！）

修改尺寸后拓扑数组下标随时会变，必须使用几何语义筛选：

```python
from build123d import Axis, GeomType, Plane, SortBy

# 1. 选取六个极值外表面
top_face    = part.faces().sort_by(Axis.Z)[-1]   # 最高顶面 (+Z)
bottom_face = part.faces().sort_by(Axis.Z)[0]    # 最低底面 (-Z)
front_face  = part.faces().sort_by(Axis.Y)[0]    # 最前正面 (-Y)
back_face   = part.faces().sort_by(Axis.Y)[-1]   # 最后背面 (+Y)
left_face   = part.faces().sort_by(Axis.X)[0]    # 最左侧面 (-X)
right_face  = part.faces().sort_by(Axis.X)[-1]   # 最右侧面 (+X)

# 2. 只选竖直方向的四条外转角边（过滤掉圆柱内孔接缝边！）
outer_z_edges = (
    part.edges()
    .filter_by(Axis.Z)
    .filter_by(lambda e: abs(e.center().X) > p.base_len_x * 0.45)
)

# 3. 方向旋转速查（切割器默认从 Z=0 顶面往 -Z 向下钻孔）：
# - 在顶面 (+Z) 向下打孔: Pos(x, y, z_top) * cutter
# - 在底面 (-Z) 向上打孔: Pos(x, y, z_bot) * Rot(180, 0, 0) * cutter
# - 在正面 (-Y) 向后(+Y)打孔: Pos(x, y_front, z) * Rot(-90, 0, 0) * cutter
# - 在背面 (+Y) 向前(-Y)打孔: Pos(x, y_back, z) * Rot(90, 0, 0) * cutter
# - 在左侧 (-X) 向右(+X)打孔: Pos(x_left, y, z) * Rot(0, 90, 0) * cutter
# - 在右侧 (+X) 向左(-X)打孔: Pos(x_right, y, z) * Rot(0, -90, 0) * cutter
```

> **方向自检口诀**：右手法则或每次打完侧面孔后，直接看 `preview_sheet.png` 的 **⑨ X-Ray 透视图**，确认沉头大口朝外、细轴朝内！

---

## 三、 修改与参考现有文件的三种标准代码模板

### 模板 1：直接修改现有 `.step` 实体（填旧孔 + 打新孔 + 加安装耳）
```python
from pathlib import Path
from build123d import Align, Box, Cylinder, Pos, import_step
from cad_kit import FeatureRegistry, ToleranceProfile, make_counterbore_cutter

def build(params=None, mode="fdm"):
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry("modified_bracket", mode=mode)

    base_step = Path(__file__).parent / "refs" / "original.step"
    part = import_step(str(base_step))

    # A. 如需“移动旧孔”：先用同位置实心圆柱把旧孔填平，再在新坐标打孔！
    plug = Pos(-12.0, 0.0, 0.0) * Cylinder(radius=3.0, height=5.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = part + plug

    # B. 在新坐标打沉头孔
    new_pos = reg.mark("H1", (-15.0, 0.0, 5.0), "外移3mm后的新安装孔", kind="hole", spec="M3")
    part = part - Pos(*new_pos) * make_counterbore_cutter(tol, "M3", depth=10.0)
    return part, reg, p
```

### 模板 2：围绕现有 `.step` / `.stl` 设计配合支架（含防干涉自动校验）
```python
from pathlib import Path
from cad_kit import FeatureRegistry, ToleranceProfile

def build(params=None, mode="fdm"):
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry("motor_mount", mode=mode)

    ref_file = Path(__file__).parent / "refs" / "module.step"
    reg.add_ref(str(ref_file))  # 自动开启：(1) 渲染半透明参考件叠影 (2) 三维碰撞干涉校验！

    gap = tol.clearance("snug") # 自动按 fdm(0.12mm) / cnc(0.02mm) / exact(0mm) 留装配间隙
    # ... 建模包裹或托举 ref_file 的支架，避让空间尺寸 = 原件尺寸 + 2 * gap ...
    return part, reg, p
```

### 模板 3：对只有 `.stl` 的复杂网格直接做水密开孔/加底座（`modify_mesh`）
```python
from pathlib import Path
import trimesh
from build123d import Align, Box, Pos
from cad_kit import FeatureRegistry, ToleranceProfile, make_hole_cutter, modify_mesh

def build(params=None, mode="fdm") -> tuple[trimesh.Trimesh, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry("housing_mod", mode=mode)

    src_stl = Path(__file__).parent / "refs" / "housing.stl"
    ear = Pos(0, -25, 0) * Box(30, 10, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    hole_pos = reg.mark("H_EAR", (0, -25, 4), "新增前耳片安装孔", kind="hole", spec="M3")
    cutter = Pos(*hole_pos) * make_hole_cutter(tol, "M3", depth=10)

    out_mesh = modify_mesh(src_stl, add_parts=[ear], cut_parts=[cutter])
    return out_mesh, reg, p
```
