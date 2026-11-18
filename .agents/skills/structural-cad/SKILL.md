---
name: structural-cad
description: >-
  创建、逆向测绘、参考修改和迭代 STEP/STL 机械与 3D 打印结构件的完整工程工作流。
  当用户要求画结构件/支架/外壳/连接件、读取或修改现有 .step/.stl/.3mf 文件、预览六视图/剖面图、
  切换 FDM 打印或 CNC 公差、或提出结构尺寸修改意见时必须激活此技能。
---

# Structural CAD 技能工作流指南

本技能依托工作区内置的 `cad_kit` 库与 `uv run cad-cli` 工具链，提供 **STEP 精确实体 + STL 水密网格** 双输出、**10 视图工程图 + 红绿体积 Diff 对比图**、**逆向截面测绘** 与 **交互式 3D 剖切预览**。

---

## 一、 常用 CLI 命令速查（在 `cad-studio` 仓库根目录下执行）

```bash
# 1. 初始化新零件目录 projects/<name>/part.py（可选拷入参考模型）
uv run cad-cli init-part <name> --mode fdm --ref /path/to/ref.step

# 2. 构建零件：导出 STEP + STL + 干涉校验 + 10视图 preview_sheet.png + 修改对比 diff_sheet.png + 3D网页 viewer.html
uv run cad-cli build projects/<name>/part.py
uv run cad-cli build projects/<name>/part.py --mode cnc   # 一键临时切换 CNC 公差构建

# 3. 逆向体检任意现有 STEP / STL / 3MF（输出平面高度、孔位坐标直径、4张带毫米标尺的截面图）
uv run cad-cli inspect <file.step|file.stl> --preview

# 4. 独立对比任意两个新旧模型的三维布尔差异（绿色新增 / 红色切除）
uv run cad-cli diff old.stl new.stl -o out/diff_sheet.png

# 5. 一键在 BambuStudio 中打开 STEP 或 STL 准备切片
uv run cad-cli open-slicer projects/<name>/out/<name>.step
```

---

## 二、 三大标准作业流程 (SOP)

### 流程 A：从零创建新结构件
1. **明确约束**：确认关键外部尺寸、配合件接口（螺丝规格、孔距）、以及制造方式（`fdm` 3D打印 / `cnc` 机加工 / `exact` 名义尺寸）。若细枝末节未指定（如倒角大小、普通壁厚），采用合理工程默认值并在回复中说明，不反复追问琐碎问题。
2. **初始化与参数化编写**：
   - 运行 `uv run cad-cli init-part <part_name> --mode <fdm|cnc|exact>`。
   - 阅读参考文档：
     - 紧固件、热熔螺母、卡扣与公差标准：[fasteners_and_tolerances.md](./references/fasteners_and_tolerances.md)
     - `build123d` 防崩溃拓扑选择与建模范式：[build123d_cookbook.md](./references/build123d_cookbook.md)
   - 在 `Params` 中定义全部可调尺寸，使用 `cad_kit` 标准特征器（`make_counterbore_cutter`、`make_heatset_cutter`、`make_nut_trap_cutter`、`make_slot_cutter`、`make_rib`、`safe_fillet`、`safe_chamfer`），并用 `reg.mark(...)` 给每个孔/槽/柱打标（`H1`, `S1`, `B1`...）。
3. **构建与多模态视觉自检**：
   - 运行 `uv run cad-cli build projects/<part_name>/part.py`。
   - 调用 `read_file` 检查 `projects/<part_name>/out/preview_sheet.png`（核对正交六视图尺寸、X-Ray 沉头方向、橙色中心剖面壁厚）。
4. **交付多视图工程图、交互式 3D 预览与文件链接**：
   - 清晰提供生成的 `.step`、`.stl`、`out/preview_sheet.png`、`out/diff_sheet.png` 以及零依赖 WebGL 实时剖切交互页 `out/viewer.html` 的可点击链接（需要侧边栏内嵌交互组件时，也可将轻量 HTML 写入 `<Artifact Directory Path>` 并标注 `UserFacing=True`）。
   - 在回复中附上当前关键特征 ID 索引表，方便用户下一轮直接点名修改。

### 流程 B：参考或修改现有 STEP / STL 文件
1. **逆向测绘先行**：
   - 将原文件放入 `projects/<part_name>/refs/`，执行 `uv run cad-cli inspect refs/<file> --preview`。
   - 用 `read_file` 查看 `*_inspect.md`、`*_inspect_slices.png` 和 `*_preview_sheet.png`。重点记录：**包围盒 `min/max` 原点位置**、**安装孔圆心 `(x, y, z)` 与直径**、**台阶高度 `Z`**。
2. **按场景选择最佳修改技术路线**：
   - **路线 1：原文件是 `.step` 且做局部增减孔/槽/耳片** → 在 `part.py` 中 `base = import_step(...)`，直接在精确坐标上进行 `build123d` 布尔加减，继续输出原生纯净 `.step` + `.stl`！
   - **路线 2：原文件是复杂/曲面 `.stl`，只需局部打孔、切槽、加底座** → 使用 `cad_kit.modify_mesh(ref_stl, add_parts=[...], cut_parts=[...])` 借助 `manifold3d` 做毫秒级水密网格手术，输出水密 `.stl`（并在构建时对比原 STL 生成 `diff_sheet.png`，把原 STL 预先拷一份到 `out/.prev/<part_name>_prev.stl` 即可自动出对比图！）。
   - **路线 3：原文件是规则机械件 `.stl`，但用户希望大改尺寸或导出真 `.step`** → 根据 `inspect` 测出的截面尺寸与孔位，用 `build123d` **1:1 逆向参数化重构**，一步升级为可调参的纯净 STEP 源码！
   - **路线 4：围绕参考件设计配合支架/外壳** → 在 `part.py` 中调用 `reg.add_ref(str(ref_path))`。构建时会自动渲染**半透明参考原件叠影**，并自动运行**三维干涉碰撞检测**（确保留有 `tol.clearance("snug"|"slide")` 配合间隙，零碰撞）。

### 流程 C：听懂口语化修改意见并极速迭代
当用户看着预览图提出修改（如 *“把正面槽往左挪 3mm 加深一点，H1 改 M4 热熔螺母，背面加两条加强筋”*）：
1. **方位核对**：严格对照 `AGENTS.md` 方位表（前=`-Y`、后=`+Y`、左=`-X`、右=`+X`、上=`+Z`、下=`-Z`）以及 `manifest.json` 中 `H1` 的当前坐标。
2. **参数级精准修改**：优先修改 `Params` 字段或对应特征块，保持其余特征锚点稳定。
3. **强制 Diff 校验**：运行 `uv run cad-cli build projects/<part_name>/part.py`，**必须调用 `read_file` 查看生成的 `out/diff_sheet.png`**：
   - 确认 🟢 **绿色新增块** 和 🔴 **红色切除块** 出现在完全正确的面和方向上！如果发现移反了轴向，立即修正后再回复用户。
4. 更新侧边栏 `viewer.html` Artifact（此时 3D 预览器右上角会自动多出 `+新增结构` 和 `-切除部分` 图层开关！）。
