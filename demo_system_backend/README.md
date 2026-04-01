## demo_system_backend

`demo_system_backend` 是对原始 `demo_system` 的**后端精简版**，去除了所有 PyQt 前端 UI，仅保留检测与可视化逻辑，用于对整段离线视频进行推理并输出带标注的 `output.mp4`。

> **建议使用无遮挡、人物清晰可见的视频作为 demo sample。** 遮挡会导致人体关键点置信度下降，tracking ID 频繁切换，影响 action 识别和 anomaly 检测的准确性。

---

### 与 demo_system 的区别

| | demo_system | demo_system_backend |
|---|---|---|
| **前端** | PyQt5 桌面 GUI | 无 GUI，纯终端 |
| **输入** | 实时摄像头 / UI 选择视频 | `--input` 指定 mp4 路径 |
| **输出** | 窗口实时显示 | 写出 `output.mp4` |
| **调度** | `DetectionProcessor(QObject)` + 线程池 | `pipeline.py` 5 个同步纯函数 |
| **依赖** | PyQt5, pygame 等 | 不需要 PyQt5, pygame |
| **MMAction2** | 需要外部安装 mmcv/mmengine/mmaction/mmdet | 已本地化至 `mmlab_local/`，无需额外安装 |

---

### 目录结构

```
demo_system_backend/
├── main.py                  # 入口：参数解析、视频读写、驱动 pipeline loop
├── pipeline.py              # 5 个 pipeline 阶段函数 + PipelineState
├── draw_utils.py            # 纯 OpenCV 可视化（骨架、框、标签）
├── run_backend.sh           # bash 启动脚本（所有参数显式定义）
├── models/
│   ├── base_detector.py     # 检测器抽象基类
│   ├── pose_detector.py     # YOLO Pose + BoT-SORT tracking
│   └── action_recognizer.py # VideoMAE 动作识别 + PCA + label mapping
├── mmlab_local/             # 本地化的 OpenMMLab 组件（无需外部 mmcv/mmaction）
│   ├── backbone_vit.py      # VideoMAE ViT-Large backbone
│   ├── detector.py          # FastRCNN (inference-only)
│   ├── roi_head.py          # AVARoIHead
│   ├── bbox_head.py         # BBoxHeadAVA
│   ├── roi_extractor.py     # SingleRoIExtractor3D (torchvision RoIAlign)
│   ├── model_builder.py     # 直接构建模型，绕过 mmengine registry
│   ├── model_config.py      # 硬编码的模型架构参数
│   ├── label_utils.py       # AVA label map 加载 + top-k action 解析
│   ├── ava_label_map.txt    # 80 类 AVA 动作标签（1-indexed）
│   ├── structures.py        # InstanceData / ActionDataSample
│   ├── base_module.py       # 轻量 BaseModule (nn.Module wrapper)
│   ├── cnn_bricks.py        # build_norm_layer, DropPath, FFN, PatchEmbed
│   └── image_utils.py       # rescale_size, imresize, imnormalize_
├── temp_requirements.txt    # Python 依赖
└── README.md
```

---

### Pipeline 设计

`main.py` 的 for-loop 对每一帧依次调用 5 个函数，全部定义在 `pipeline.py`，完全同步、线性执行：

```
for frame_idx in range(total_frames):
    frame          = preprocess(raw_frame)
    det_results    = detection(frame, state)
    action_results = action(frame, det_results, state)
    bio            = biomarker(action_results, state)
    annotated      = postprocess(frame, det_results, bio)
```

#### 各阶段说明

| # | 阶段 | 输入 | 输出 | 说明 |
|---|------|------|------|------|
| 1 | **preprocess** | `raw_frame`: BGR uint8 (原始分辨率) | `frame`: BGR uint8 (长边 ≤ max_width) | 等比缩放，限制分辨率 |
| 2 | **detection** | `frame`: BGR uint8 (H, W, 3) | `det_results`: dict — `keypoints` (N,17,2), `scores` (N,17), `boxes` (list of xyxy), `track_ids` (list of int), `count` (int) | YOLO pose `model.track()` 同时完成检测 + BoT-SORT 跟踪，只保留置信度最高的 `top_identity` 个人 |
| 3 | **action** | `frame`, `det_results`, `state` (滑动窗口 buffer) | `action_results`: dict or None | 每 5 帧触发一次，从 buffer 中采样 16 帧送入 VideoMAE FastRCNN；按 identity 分别做 81 类 AVA 动作分类 |
| 4 | **biomarker** | `action_results`, `state` (per-identity 历史) | `bio`: dict or None | 对每个 identity 的 PCA scalar 做滑动窗口方差计算，超过 `anomaly_threshold` 标记为异常 |
| 5 | **postprocess** | `frame`, `det_results`, `bio` | `annotated`: BGR uint8 | 画骨架、bounding box、`P{id}` 标签、anomaly "!" 标记、全局 ANOMALY 横幅 |

---

### DELIVER

每帧处理完毕后，`main.py` 会打包一个 `DELIVER` 字典，汇总当前帧的所有输出，供未来下游任务使用：

```python
DELIVER = {
    "frame_idx":        int,           # 原始视频中的帧序号
    "det_results":      dict,          # detection 输出（boxes, keypoints, track_ids, ...）
    "action_results":   dict or None,  # action 输出，每个 identity 包含：
    #   identity_results[track_id] = {
    #       "raw_scores":  np.ndarray (81,)    — 81 类 AVA 动作的 sigmoid logits
    #                                            index 0 = background, 1-80 = 动作类
    #       "top_actions": [(name, score), ...] — top-3 动作名 + 分数
    #                                            name 从 ava_label_map.txt 查得
    #                                            (1-indexed class_id → 动作名)
    #       "pca_scalar":  float or None        — 81-d score → PCA 降维后的标量
    #       "scores":      float                — 同 pca_scalar，或 0
    #       "num_frames":  int                  — 用于推理的帧数
    #       "boxes":       np.ndarray (K, 4)    — 该 identity 的 bounding boxes
    #   }
    "biomarker_summary": dict or None, # biomarker 输出，每个 identity 包含：
    #   identities[track_id] = {
    #       "pca_scalar":  float or None
    #       "scores":      float
    #       "variance":    float          — 最近 5 次 pca_scalar 的归一化方差
    #       "anomaly":     bool           — variance > anomaly_threshold
    #       "top_actions": [(name, score), ...]  — 同上，方便下游直接使用
    #   }
    "annotated":        np.ndarray,    # BGR uint8，带标注的可视化帧
}
```

---

### 使用方式

#### 通过 bash 脚本（推荐）

编辑 `run_backend.sh` 顶部的变量，然后执行：

```bash
cd demo_system_backend
bash run_backend.sh
```

`run_backend.sh` 中的可配置变量：

```bash
INPUT_VIDEO="..."              # 输入视频路径
OUTPUT_VIDEO="..."             # 输出视频路径
DEVICE="cuda:0"                # 计算设备
YOLO_MODEL="..."               # YOLO pose 模型路径 (.pt)
ACTION_CHECKPOINT="..."        # VideoMAE action 模型路径 (.pth)
FRAME_STEP=8                   # 抽帧步长（每 N 帧取 1 帧）
TOP_IDENTITY=1                 # 只追踪置信度最高的 N 个人
ANOMALY_THRESHOLD=0.145        # 异常检测方差阈值
```

#### 直接执行

```bash
cd demo_system_backend

python main.py \
    --input       input.mp4 \
    --output      output.mp4 \
    --device      cuda:0 \
    --yolo_model  /path/to/yolo11x-pose.pt \
    --checkpoint  /path/to/videomae_ava.pth \
    --frame_step  8 \
    --top_identity 1 \
    --anomaly_threshold 0.145
```

#### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input`, `-i` | (必填) | 输入视频路径 |
| `--output`, `-o` | `output.mp4` | 输出视频路径 |
| `--device` | `cuda:0` | 计算设备 |
| `--mode` | `balanced` | 模型模式 (lightweight / balanced / performance) |
| `--max_width` | `640` | 输出帧最大宽度 |
| `--yolo_model` | `backend_weights/yolo11n-pose.pt` | YOLO pose 模型路径 |
| `--checkpoint` | `None` | VideoMAE action 模型 checkpoint 路径 |
| `--frame_step` | `8` | 抽帧步长（每 N 帧保留 1 帧，输出帧率 = 原帧率 / N） |
| `--top_identity` | `1` | 每帧只追踪置信度最高的 N 个人 |
| `--anomaly_threshold` | `0.145` | biomarker 方差异常阈值 |

---

### 环境搭建

#### 1. 创建 conda 环境 + 安装 PyTorch

以下版本与开发服务器完全一致：

| 包 | 版本 |
|---|---|
| Python | 3.9.13 |
| torch | 2.1.0 (CUDA 12.1) |
| torchvision | 0.16.0 |

```bash
conda create -n demo_backend python=3.9.13 -y
conda activate demo_backend
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 2. 安装 pip 依赖

```bash
cd demo_system_backend
pip install -r requirements.txt
```

不需要安装 mmcv / mmengine / mmaction2 / mmdet — 相关代码已本地化至 `mmlab_local/`。
