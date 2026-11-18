# 3D 打印可制造性 (DFM) 体检、自动寻优摆放与多件总装 (Assembly) 指南

本手册说明如何使用 `v0.3` 新增的 **FDM 悬垂/孤岛检测与自动最佳摆放 (`print-check`)**、**原生多对象彩色 `.3mf` 导出**，以及 **多零件总装系统 (`Assembly` + 爆炸视图 + BOM 备料表)**。

---

## 一、 3D 打印可制造性体检与 6 向最佳摆放 (`cad-cli print-check`)

当设计复杂异形支架、卡扣、倒扣件，或准备将零件送入切片软件打印前：

```bash
# 1. 对任意 part.py / .step / .stl 运行悬垂热力与半空孤岛扫描，评估 6 个贴床朝向得分
uv run cad-cli print-check projects/<name>/part.py

# 2. 自动旋转至第 1 名最佳免支撑朝向，导出 *_oriented.step 与 *_oriented.3mf
uv run cad-cli print-check projects/<name>/part.py --auto-orient

# 3. 在常规 build 时同步附上 DFM 热力报告图 (out/printability_sheet.png)
uv run cad-cli build projects/<name>/part.py --print-check
```

### 报告解读与修正准则
- **半空无支撑孤岛 (`floating_islands_count > 0`)**：指在热床以上某高度突然凭空出现的局部最低点，无支撑直接打印 100% 炒面！优先采用 `best_orientation` 换向消除，或在设计上将悬空柱延伸落地 / 改为 `45°` 斜撑过渡。
- **重度悬垂 (`severe_overhang_mm2` > 55°)**：若无法通过换向消除，提醒用户在切片软件开启树状支撑，或将圆角底边改为 `45°` 倒角。
- **小底面高瘦件 (`needs_brim=True`)**：提醒用户在 BambuStudio 开启 `5~8mm` 外侧 Brim 裙边。

---

## 二、 多零件总装编排 (`Assembly` + 爆炸图 + 紧固件 BOM + 自动排盘 `.3mf`)

当项目包含多个配合零件（如 `下壳 + 上盖 + 导轨卡座 + 按键 + 参考PCB/电机`）时，使用总装工作流：

```bash
# 1. 初始化多件总装模板 projects/<asm_name>/assembly.py
uv run cad-cli init-assembly my_device_unit

# 2. 构建总装（自动执行 N×N 全矩阵防撞与间隙检测 + 6视图彩色爆炸图 + BOM表 + 自动排盘打印包 .3mf + 带爆炸滑块的 viewer.html）
uv run cad-cli build-assembly projects/my_device_unit/assembly.py
```

### `assembly.py` 编写范式
```python
from cad_kit import Assembly, ToleranceProfile

def build_assembly(mode: str = "fdm") -> Assembly:
    tol = ToleranceProfile.for_mode(mode)
    asm = Assembly(name="my_device", description="多件总装示例", mode=mode)

    # 1. 添加自制结构件（自动分配高对比工程配色，自动参与打印排盘 .3mf）
    asm.add_part("base_shell", base_part, label="主下壳", explode_vec=(0, 0, 0), registry=reg_base)
    asm.add_part("top_cover", lid_in_assembled_pose, label="防尘上盖", explode_vec=(0, 0, 40.0))

    # 2. 添加外购/参考件（is_reference=True：半透明显示、参与防撞检测、自动排除在打印排盘 .3mf 之外！）
    asm.add_part("pcb_mod", "refs/board.step", label="主控PCB", is_reference=True, explode_vec=(0, 0, 20.0))

    # 3. 登记紧固件与标准件采购 BOM
    asm.add_hardware("M3×8 内六角圆柱头螺丝", qty=4, category="紧固件", note="锁紧上下盖")
    asm.add_hardware("M3×4.5×5 热熔滚花铜螺母", qty=4, category="嵌件", note="热压入下壳四角")
    return asm
```

### 总装自动交付的 6 项核心产物 (`out/`)
1. **`assembly_sheet.png`**：含闭合态等轴测、**3D 爆炸拆解图**、**彩色多件中心实心剖视图**（核验止口咬合与内部间隙！）及正/右/顶视图；
2. **`bom.md`**：自动计算每件自制件体积与**预估打印克重（g）**、汇总紧固件清单、列出 **N×N 零件两两干涉体积与实测最小表面间隙（mm）**；
3. **`<name>_print_plate.3mf`**：自动剔除外购参考件，将所有自制打印件**逐一自动寻优翻转至最佳免支撑朝向并平铺在 `Z=0` 热床**，拖入 BambuStudio 即可直接切片！
4. **`<name>_assembly.3mf` & `<name>_assembly.step`**：装配闭合态多对象命名彩色工程包；
5. **`viewer.html`**：自动激活 **`💥 装配体动态爆炸拆解 (0% ~ 100%)`** 滑块，且支持在拆解态下继续使用 **🎯 射线点选改件** 与 **📏 3D 两点卡尺**！
