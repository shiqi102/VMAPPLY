# VMAPPLY / DriveWorld

基于 HuggingFace 世界模型（CogVideoX-5B、Stable Video Diffusion）的**自动驾驶长尾场景生成与风险评估**系统。面向雨夜、路口、cut-in、行人横穿、急刹等复杂驾驶场景，生成具有时序连续性的驾驶视频，并通过 SegFormer + SAM2 视觉解析、光流跟踪与 TTC/PET 等指标完成场景风险评估。

> 仓库地址：[github.com/shiqi102/VMAPPLY](https://github.com/shiqi102/VMAPPLY)

## 系统架构

```
文本/关键帧 → [CogVideoX / SVD] → 质量筛选 → 生成视频
                                              ↓
                                    [SegFormer 道路语义分割]
                                              ↓
                              [SAM2 基于 bbox 的精细实例分割]
                                              ↓
                              [Farneback 光流 + IoU 多目标跟踪]
                                              ↓
                         [TTC / 车道侵入 / 遮挡 / 轨迹冲突 风险评估]
                                              ↓
              风险分数 + 热力图 + 关键帧 + 分割/跟踪可视化
```

## 功能模块

| 模块 | 技术栈 | 功能 |
|------|--------|------|
| 视频生成 | CogVideoX-5B, SVD-XT | 文本到视频、关键帧到视频 |
| 质量筛选 | Laplacian 锐度 + 时序一致性 | 多轮生成选优 |
| 道路解析 | SegFormer-B5 (Cityscapes 19 类) | 可行驶区域、人行道、道路边缘、动态目标粗检 |
| 目标分割 | SAM2-Hiera-Large | 基于 SegFormer bbox 的精细 mask |
| 运动建模 | Farneback 光流 + IoU 跟踪 | 目标轨迹、自车运动估计 |
| 风险评估 | TTC, 车道侵入, 遮挡, 轨迹冲突 | 风险分数、热力图、关键风险帧 |
| 可视化 | ParsingVisualizer | SegFormer / SAM2 / 跟踪分割效果图 |

## 环境配置

```bash
# 推荐使用 conda
conda create -n worldmodel python=3.10 -y
conda activate worldmodel

cd VMAPPLY   # 或你的克隆目录
pip install -r requirements.txt

# 可选：安装原生 SAM2（未安装时自动回退到 HuggingFace transformers 接口）
pip install git+https://github.com/facebookresearch/sam2.git
```

**硬件建议**

| 模块 | 显存 |
|------|------|
| CogVideoX-5B 文生视频 | ≥ 16 GB |
| SVD 图生视频 | ≥ 8 GB |
| SegFormer + SAM2 解析 | ≥ 8 GB |

## 模型权重下载

首次运行会从 Hugging Face 拉取权重。国内环境建议：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 按需下载（支持断点续传）
hf download THUDM/CogVideoX-5b --max-workers 1
hf download stabilityai/stable-video-diffusion-img2vid-xt --max-workers 1
hf download nvidia/segformer-b5-finetuned-cityscapes-1024-1024
hf download facebook/sam2-hiera-large
```

| 模型 | 用途 | 约大小 |
|------|------|--------|
| `THUDM/CogVideoX-5b` | 文生视频 | ~21 GB |
| `stabilityai/stable-video-diffusion-img2vid-xt` | 图生视频 | ~9–31 GB |
| `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` | 道路解析 | ~300 MB |
| `facebook/sam2-hiera-large` | 目标分割 | ~1–2 GB |

长时间下载建议放在 tmux 中执行，避免 SSH 断连中断任务。

## 快速开始

### 文本到视频（CogVideoX）+ 全流程分析

```bash
conda activate worldmodel

# 使用预设场景 cut_in，seed 默认读取 config generation.seed
python main.py --mode text2video --scenario cut_in

# 自定义提示词 / 指定种子
python main.py --mode text2video --scenario pedestrian_crossing --seed 42
python main.py --mode text2video --prompt "A rainy night highway with sudden braking ahead"
```

### 关键帧到视频（SVD）+ 全流程分析

```bash
# 使用 config 中已配置 keyframe 的场景
python main.py --mode image2video --scenario kitti_test

# 指定关键帧路径
python main.py --mode image2video \
  --scenario kitti_test \
  --keyframe data/kitti/images/0000000018.png \
  --seed 42
```

> SVD 模式需要场景配置 `keyframe` 或通过 `--keyframe` 传入图片；仅配置了 `text_prompt` 的场景（如 `cut_in`）不能直接用于图生视频。

### 已有视频分析（跳过生成）

```bash
python main.py --mode analyze \
  --video outputs/cut_in/generated.mp4 \
  --scenario cut_in
```

### 批量生成所有预设场景

```bash
python scripts/generate_scenarios.py
```

## 预设长尾场景

场景定义在 `config/default.yaml` 的 `scenarios` 中：

| 场景 | 描述 | 文生视频 | 图生视频 |
|------|------|:--------:|:--------:|
| `rainy_night` | 雨夜湿滑路面 | ✓ | 需配置 keyframe |
| `intersection` | 繁忙城市路口 | ✓ | 需配置 keyframe |
| `cut_in` | 高速突然变道切入 | ✓ | 需配置 keyframe |
| `pedestrian_crossing` | 行人突然横穿 | ✓ | 需配置 keyframe |
| `emergency_brake` | 前车急刹 | ✓ | 需配置 keyframe |
| `kitti_test` | KITTI 前视示例 | ✓ | ✓（已配 keyframe） |

```yaml
scenarios:
  cut_in:
    description: "高速公路突然变道切入"
    text_prompt: "A dashcam view on a highway, ..."
    keyframe: null

generation:
  seed: 42   # 默认随机种子，可用 --seed 覆盖
```

## 输出结构

以 `cut_in` 场景为例，完整跑通后输出如下：

```
outputs/cut_in/
├── generated.mp4              # 生成的驾驶视频（49 帧 @ 8fps）
├── result.json                # 汇总：解析统计、轨迹、风险等级
├── risk_report.json           # 详细风险评估报告
├── risk_viz/
│   ├── risk_timeline.png      # 逐帧风险曲线
│   ├── risk_heatmap.png       # 风险热力图
│   └── key_frame_*.png        # 高风险关键帧原图
└── parsing_viz/               # save_intermediate: true 时生成
    ├── legend_classes.png     # Cityscapes 19 类图例
    ├── parsing_overview.mp4   # 分割+跟踪三行拼图视频
    ├── overview_*.png         # 单帧总览（SegFormer / SAM2 / Tracking）
    ├── segformer/             # 道路语义 + 可行驶区域 + 边缘
    ├── sam2/                  # 精细目标 mask
    └── tracking/              # IoU 跟踪 ID + 轨迹线
```

### 如何解读结果

**`result.json` 示例字段：**

```json
{
  "parsing_summary": {
    "total_objects_detected": 284,
    "object_classes": ["car", "truck", "bus", "person"],
    "avg_road_coverage": 0.39
  },
  "trajectory_summary": { "num_tracks": 14 },
  "risk_assessment": {
    "overall_score": 0.128,
    "level": "low",
    "conflicts": 14
  }
}
```

**分割可视化说明：**

| 可视化 | 含义 |
|--------|------|
| SegFormer 绿色区域 | `road` 可行驶区域 |
| SegFormer 紫色区域 | `sidewalk` 人行道 |
| SegFormer 青色线 | 可行驶区域 Canny 边缘（道路边界近似，非真实车道线） |
| SAM2 彩色 mask | 在 SegFormer bbox 上的精细实例分割 |
| Tracking `ID:N` | IoU 跨帧关联的跟踪 ID（非 SAM2 内置功能） |

## 项目结构

```
VMAPPLY/
├── config/default.yaml        # 场景、模型、风险权重、可视化配置
├── driveworld/
│   ├── generation/            # CogVideoX, SVD, 质量筛选
│   ├── parsing/               # SegFormer, SAM2, 可视化
│   ├── motion/                # 光流、轨迹、IoU 跟踪
│   ├── risk/                  # TTC、侵入、冲突评估
│   └── pipeline.py            # 端到端流程
├── scripts/
│   ├── generate_scenarios.py  # 批量生成
│   ├── parse_scene.py         # 独立场景解析
│   └── assess_risk.py         # 独立风险评估
├── data/kitti/images/         # 示例关键帧
├── main.py                    # 主入口
└── requirements.txt
```

## 配置说明

`config/default.yaml` 主要配置项：

| 配置路径 | 说明 |
|----------|------|
| `scenarios.*.text_prompt` | 文生视频提示词 |
| `scenarios.*.keyframe` | 图生视频关键帧路径 |
| `generation.seed` | 默认随机种子 |
| `generation.cogvideox` | 帧数、步数、guidance scale |
| `generation.svd` | motion bucket、noise augmentation |
| `generation.quality_filter` | 锐度/亮度/时序一致性阈值 |
| `output.save_intermediate` | 是否保存分割可视化 |
| `output.parsing_viz.max_frames` | 可视化采样帧数（默认 10） |
| `risk.weights` | TTC、侵入、遮挡等权重 |

## 常见问题

**Q: `SentencePiece` / tokenizer 报错？**

```bash
pip install sentencepiece
```

**Q: HuggingFace 下载 SSL 中断或卡在 0%？**

使用镜像并关闭代理与 xet（见上方「模型权重下载」）。

**Q: 只想重新跑分析，不重新生成视频？**

使用 `--mode analyze --video outputs/<scenario>/generated.mp4`。

## License

MIT（如适用请根据实际需求调整）
