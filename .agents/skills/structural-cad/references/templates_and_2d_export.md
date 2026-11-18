# 结构母版库 (`cad_kit.templates`)、二维 DXF/SVG 导出与 3D 点选口令解析指南

当用户需要创建**电子外壳、铝型材连接件、DIN 导轨卡座、齿轮/轴承座**，或需要导出 **1:1 激光切割/CNC 加工图（DXF/SVG）**，或从 3D 预览器粘贴了 **点选/卡尺口令** 时，优先参考本指南。

---

## 一、 一键脚手架模板 (`cad-cli init-part --template`)

```bash
# 1. 上下分体子母止口电子外壳（含四角热熔螺母柱、PCB支撑柱、侧散热百叶、挂墙耳、USB-C避让口）
uv run cad-cli init-part my_box --template enclosure

# 2. 2020 / 3030 / 4040 欧标铝型材防转直角加强角码（自带双面T型防转卡舌）
uv run cad-cli init-part corner_2020 --template extrusion-bracket

# 3. 35mm 欧标 DIN 导轨 (TS35) 弹性快拆卡扣底座（自带悬臂弹片与一字起撬口）
uv run cad-cli init-part plc_clip --template din-clip

# 4. 渐开线正齿轮（真渐开线齿廓 + 可选 D 型轴孔 + 凸台顶丝防转螺母槽）
uv run cad-cli init-part drive_gear --template gear
```

---

## 二、 代码级母版 API 速查 (`from cad_kit import ...`)

### 1. 分体式电子外壳 (`make_split_enclosure` + `cut_panel_port`)
```python
from cad_kit import EnclosureParams, make_split_enclosure, cut_panel_port

cfg = EnclosureParams(
    outer_x=90.0, outer_y=62.0, bottom_h=24.0, lid_h=8.0,
    wall_t=2.8, corner_r=5.0, lip_h=2.2,
    flange_ears=True,             # 左右挂墙安装耳
    vent_slots=5,                 # 侧壁散热格栅
    pcb_standoffs=[(-30, -20), (30, -20), (-30, 20), (30, 20)],
    pcb_standoff_h=5.0,
    pcb_screw="M2.5",
)
# layout 可选:
# - 'print_plate': 下壳与翻平上盖并排铺在 Z=0 热床，100% 免支撑直接切片！
# - 'assembled'  : 上盖扣合在下壳上方（可通过 explode_z=12.0 生成上下爆炸展示图）
# - 'bottom' / 'lid': 单独导出下壳或上盖
part = make_split_enclosure(cfg, tol=tol, reg=reg, layout="print_plate")

# 一行代码在任意侧壁 (-Y/+Y/-X/+X) 开标准接口（usb_c / dc_jack / sma / rect / circle）
part = cut_panel_port(part, wall="-Y", port_type="usb_c", offset_along_wall=-12.0, center_z=10.5, tol=tol, reg=reg)
```

### 2. 欧标铝型材与 35mm DIN 导轨组件
- `make_tslot_tongue(series="2020"|"2040"|"3030"|"4040", length=30.0, tol=tol)`：生成梯形防转卡舌（默认在 Z=0 向 -Z 凸出），合并到任意支架背面即可卡入型材槽防旋转。
- `make_extrusion_bracket(series="2020", arm_len=44.0, with_tongues=True, tol=tol, reg=reg)`：双面防转直角角码。
- `make_extrusion_mount_plate(series="2020", length_x=56.0, width_y=36.0, tol=tol, reg=reg)`：型材设备转接板。
- `make_din_rail_clip(width_x=18.0, top_deck_y=48.0, mount_screw="M3", tol=tol, reg=reg)`：标准 35mm TS35 导轨卡扣（提醒用户切片时侧放 `-X/+X` 贴热床打印，使弹片沿层纹不断裂）。

### 3. 齿轮、电机 D 轴与轴承座
- `make_spur_gear(module=1.5, teeth=24, thickness=6.0, bore_d=5.0, d_shaft_flat=4.5, hub_d=16.0, hub_h=7.0, set_screw="M3", tol=tol, reg=reg)`
- `make_d_shaft_cutter(shaft_d=5.0, flat_d=4.5, depth=20.0, tol=tol)`
- `make_bearing_block(bearing_code="608"|"625"|"MR85"|"6000", shaft_center_z=18.0, tol=tol, reg=reg)`

---

## 三、 1:1 激光切割 / CNC 二维图纸导出 (`DXF` + `SVG` + 钻孔坐标表)

- **构建时顺带导出**：
  `uv run cad-cli build projects/<name>/part.py --dxf bottom`（可选 `bottom/top/front/back/left/right/XY/XZ/YZ`）
- **对任意现成 `.step` / `.stl` 独立导出**：
  `uv run cad-cli export-2d path/to/part.step --face bottom`
  `uv run cad-cli export-2d path/to/part.stl --plane XY --offset 2.0`
- **技术特性**：自动避开底面 `0.4mm` 防象脚倒角以获取真实名义外轮廓；自动分层 `INNER_HOLES`（青色，激光先割内孔）与 `OUTER_CUT`（白色/黑色，后割外框），圆孔保留原生 CAD `CIRCLE` 解析实体，并配套生成带孔位坐标表的 `*_2d.png` 核对图。

---

## 四、 解析来自 `viewer.html` 的点选口令与 3D 卡尺数据

用户在 `viewer.html` 中点击表面、锐边或使用两点卡尺后，会直接复制如下格式的指令：
1. **点选修改口令**：
   `修改零件 xxx：针对 位置 (x, y, z) mm [所在面朝向: 右侧面 Right (+X)]，在此处...`
   - `(x, y, z)` 即该特征在当前模型全局坐标系下的**精确毫米坐标**！
   - 若在此处垂直表面打孔，直接结合 `[所在面朝向]` 选用对应旋转（例如右侧面 `+X` 向左 `-X` 打孔用 `Pos(x, y, z) * Rot(0, -90, 0) * cutter`）。
2. **卡尺实测口令**：
   `零件 xxx 实测两点间距：P1(x1, y1, z1) ↔ P2(x2, y2, z2) 当前总距 = L mm (|ΔX|=..., |ΔY|=..., |ΔZ|=...)。请将该间距调整为：...`
   - 核对 `|ΔX|, |ΔY|, |ΔZ|` 判断用户测的是壁厚、孔心距还是台阶高度，定位到 `Params` 对应参数做差值修正，并用 `diff_sheet.png` 验证。
