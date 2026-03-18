## demo_system_backend

`demo_system_backend` 是对原始 `demo_system` 的**后端精简版**，去除了所有 PyQt 前端 UI，仅保留检测与可视化渲染逻辑，用于对整段离线视频进行推理并导出带可视化结果的 `output.mp4`。

### 核心功能

- **输入方式**: 仅支持从本地视频文件（例如 `input.mp4`）读取，不再支持实时摄像头。
- **检测流水线**:
  - 直接复用 `demo_system` 中的 `DetectionProcessor` 及其内部模型（如 `PoseDetector`、`ActionRecognizer` 等），保持**中间处理逻辑与原系统一致**。
  - 每帧进行姿态 / 动作相关检测，并在图像上绘制骨架与边框。
- **输出方式**:
  - 将带有骨架和边框可视化结果的视频按帧写入一个新的 MP4 文件（默认 `output.mp4`）。
  - 为了控制文件大小，输出会限制最大宽度（默认 `640` 像素），使用较低分辨率与常规编码参数。

### 与原始 demo_system 的区别

- **前端/UI**:
  - `demo_system`:
    - 使用 PyQt5，包含 `main_window.py`、`VideoDisplay`、`ControlPanel` 等完整桌面 UI。
    - 实时显示摄像头 / 视频流、控制检测模式、显示统计与日志等。
  - `demo_system_backend`:
    - 完全移除所有窗口控件、菜单栏、状态栏和交互控件。
    - 不需要也不会启动任何图形界面，只在终端输出日志。

- **输入源**:
  - `demo_system`:
    - 支持实时摄像头输入和通过 UI 选择本地视频文件。
  - `demo_system_backend`:
    - 输入视频路径通过命令行参数指定，**仅支持本地视频文件**，不再访问摄像头。

- **输出形式**:
  - `demo_system`:
    - 在窗口中实时显示检测结果，同时可用于交互式演示。
  - `demo_system_backend`:
    - 对整段视频顺序推理，将可视化叠加后的结果写入一个 `mp4` 文件（例如 `output.mp4`），适合离线批处理或生成演示视频。

- **代码结构**:
  - `demo_system`:
    - 拥有 `app/`, `ui/`, `core/`, `models/` 等完整目录，包含大量前端相关代码。
  - `demo_system_backend`:
    - 仅新增少量后端入口与绘制工具：
      - `main.py`: 后端主入口，负责读取视频、调用 `DetectionProcessor`、写出结果视频。
      - `draw_utils.py`: 从 `demo_system` 的 `VideoDisplay` 中抽取并改造成**纯 OpenCV** 的绘制函数，用于在帧上画骨架与框，无任何 UI 依赖。
      - `run_backend.sh`: 便捷的 Bash 启动脚本。

### 依赖关系

- `demo_system_backend` 直接复用 `demo_system` 中的后端逻辑，因此需要：
  - 已正确安装 `demo_system/requirements.txt` 中列出的依赖；
  - 已按 `demo_system/README.md` 下载好对应模型权重（如 YOLO、MMAction2 等）；
  - 保持 `demo_system` 目录结构不变。

### 使用方式

#### 1. 通过 Bash 脚本运行（推荐）

在仓库根目录下：

```bash
bash demo_system_backend/run_backend.sh path/to/input.mp4
```

可选地指定输出路径：

```bash
bash demo_system_backend/run_backend.sh path/to/input.mp4 path/to/output.mp4
```

#### 2. 直接运行 Python 模块

同样在仓库根目录下：

```bash
python -m demo_system_backend.main --input path/to/input.mp4 --output output.mp4
```

可选参数：

- `--device`: 模型推理设备，默认 `"cuda:0"`，若无 GPU 可设为 `"cpu"`。
- `--mode`: 与原始 `demo_system` 一致的模型模式，`lightweight | balanced | performance`，默认 `balanced`。
- `--max_width`: 输出视频的最大宽度（像素），默认 `640`，用于控制输出文件大小。

### 典型工作流

1. 按原始 `demo_system/README.md` 完成依赖安装与权重下载。
2. 将一段需要分析的老人活动监测视频如 `input.mp4` 放在本地。
3. 在仓库根目录执行：
   ```bash
   bash demo_system_backend/run_backend.sh input.mp4
   ```
4. 等待推理结束后，在当前目录或指定路径获得 `output.mp4`，其中包含与原 `demo_system` 类似的骨架和检测可视化叠加效果。

