# AnyShadow / ESP32-P4 Shadow Detection Handoff

更新时间：2026-06-27

本文档用于给新的 Codex 窗口或其他接手者快速理解本对话中的过程、当前仓库状态、可运行工具、已知问题和下一步建议。

## 1. 用户最终目标

用户想做一个能在纸面低像素画面中识别阴影的 AI，并最终部署到 ESP32-P4 上实时显示阴影轮廓。

当前活跃目标更具体：

> 跑通 AnyShadow，做到 demo 里面通过第一帧画出阴影，然后识别之后视频中的阴影。

换句话说，现在重点不是继续训练 ESP32 小模型，而是先让电脑端强 teacher 模型可用：用户在视频第一帧手动画出阴影种子，模型用 SAM/AnyShadow/LSTN 传播到后续帧，生成 overlay 视频。这个 teacher 结果后面再用于伪标签、微调和蒸馏到 P4。

## 2. 仓库和环境

当前工作目录：

```bash
/Users/kongfans/Documents/Codex/2026-06-01/ai-ai-esp32-hugging-face-plugin
```

重要 Python 环境：

```bash
work/.venv_shadow/bin/python
```

这个环境用于 Detect-AnyShadow 相关脚本。它已经安装了 torch、torchvision、opencv、numpy、fastapi、uvicorn 等依赖。

当前 Detect-AnyShadow 仓库位置：

```bash
work/external/Detect-AnyShadow
```

当前 FastAPI demo 服务在本机运行过，端口：

```text
http://127.0.0.1:7871
```

截至写本文档时，进程曾存在：

```bash
work/.venv_shadow/bin/python scripts/detect_anyshadow_fastapi_demo.py
```

如果新窗口接手，需要重新确认服务是否还在：

```bash
ps aux | rg "detect_anyshadow_fastapi_demo|7871"
```

如果不在，重新启动：

```bash
cd /Users/kongfans/Documents/Codex/2026-06-01/ai-ai-esp32-hugging-face-plugin
work/.venv_shadow/bin/python scripts/detect_anyshadow_fastapi_demo.py
```

然后浏览器打开：

```text
http://127.0.0.1:7871
```

## 3. 用户已提供的 AnyShadow 权重

用户通过百度网盘下载好了 Detect-AnyShadow 相关权重，原始下载位置包括：

```text
/Users/kongfans/Downloads/finetune.pth
/Users/kongfans/Downloads/sam_vit_b_01ec64.pth
/Users/kongfans/Downloads/lstnb/
/Users/kongfans/Downloads/lstns/
/Users/kongfans/Downloads/lstnt/
```

已复制/整理到仓库内：

```text
work/external/Detect-AnyShadow/checkpoints/chk_sam/finetune.pth
work/external/Detect-AnyShadow/checkpoints/original_sam/sam_vit_b_01ec64.pth
work/external/Detect-AnyShadow/checkpoints/lstnb/save_step_10000.pth
work/external/Detect-AnyShadow/checkpoints/lstns/save_step_10000.pth
work/external/Detect-AnyShadow/checkpoints/lstnt/save_step_10000.pth
work/external/Detect-AnyShadow/checkpoints/lstnt/save_step_60000.pth
```

当前主要使用 `lstnb/save_step_10000.pth` 做 LSTN 传播。

## 4. 为什么没有直接用原始 Gradio demo

Detect-AnyShadow 原仓库 demo 主要面向 Linux + CUDA 环境，直接在这台 Mac 上跑遇到几个问题：

- 原始 `demo_app.py` 写死 `device="cuda"`。
- LSTN 依赖 CUDA 版 `spatial_correlation_sampler`。
- 原始 Gradio 版本/API 和当前 Python 3.13 环境不完全兼容，遇到过 `audioop`、`HfFolder`、schema/self-check 等问题。

所以当前不是强行跑原始 Gradio，而是做了一个新的本地 FastAPI demo，复用 AnyShadow/SAM/LSTN 核心权重和推理逻辑。

## 5. 已修改的 Detect-AnyShadow 外部代码

为了让 Detect-AnyShadow 能在 Mac CPU/MPS fallback 下跑，修改过外部仓库内几个文件：

```text
work/external/Detect-AnyShadow/sam/build_sam.py
work/external/Detect-AnyShadow/lstn/networks/layers/transformer.py
work/external/Detect-AnyShadow/lstn/networks/models/lstn.py
```

修改目的：

- SAM 权重加载使用 `map_location="cpu"`，避免 CUDA-only 加载失败。
- LSTN 增加 `MODEL_LSAB_ENABLE_CORR` / `enable_corr` 关掉 CUDA correlation path。
- 在 CPU/MPS fallback 下绕开 `spatial_correlation_sampler`。

接手者不要随手重置 `work/external/Detect-AnyShadow`，否则 Mac 端 AnyShadow demo 可能又会不能跑。

## 6. 当前新增的关键脚本

### 6.1 本地网页 demo

```text
scripts/detect_anyshadow_fastapi_demo.py
```

这是当前最重要的脚本。它提供网页：

```text
http://127.0.0.1:7871
```

网页流程：

1. 输入视频路径、最大帧数、stride。
2. 点击“载入视频第一帧”。
3. 在第一帧 canvas 上用黄色笔刷涂阴影区域。
4. 点击“SAM 细化第一帧 mask”。
5. 点击“LSTN 传播”。
6. 页面显示并保存 `overlay.mp4`。

输出目录：

```text
work/video_shadow_compare/anyshadow_fastapi_demo/<run_id>/overlay.mp4
work/video_shadow_compare/anyshadow_fastapi_demo/<run_id>/masks/
```

已经生成过的输出包括：

```text
work/video_shadow_compare/anyshadow_fastapi_demo/253bd1035506/overlay.mp4
work/video_shadow_compare/anyshadow_fastapi_demo/831840789e00/overlay.mp4
```

### 6.2 SAM 单帧/逐帧尝试

```text
scripts/run_detect_anyshadow_sam_video.py
```

作用：用 fine-tuned SAM 从 proposal mask 做帧级 refine。

问题：速度很慢，单独使用效果也不够稳定，不适合作为最终 teacher 的主流程。

### 6.3 LSTN 视频传播

```text
scripts/run_detect_anyshadow_lstn_video.py
```

作用：给定第一帧 seed mask，用 LSTN 往后传播。

这条线是当前最接近用户想要的“第一帧画一下，后面自动识别”的方案。

### 6.4 ROI 后处理

```text
scripts/filter_teacher_masks_with_roi.py
```

作用：用动态纸面 ROI 过滤 LSTN 传播结果，避免把桌面/手/其他深色区域全部算成阴影。

效果：比完全不筛好，但还不能彻底解决手、机械臂、暗物体误检。

## 7. 当前用户视频

用户提供过的视频路径：

```text
/Users/kongfans/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_zxjeq5ptpdkt22_d12e/temp/RWTemp/2026-06/9849b8961b4b8ab8a44aef4acacc7eba/c9be5e96759298bd43cce0bb156c07e9.mp4
```

这是机械臂/手/纸面场景视频。之前全量处理结果很差，用户指出：

- 第一帧确实有一大片纸面阴影。
- 初期还能捕捉到物体/阴影。
- 随着光角度和阴影移动，框选/轮廓逐渐跑偏。
- 用户倾向先做“单一场景过拟合”或强 teacher 伪标注。

## 8. 当前 AnyShadow 结果判断

重要结论：

- 当前自训练 tiny 模型效果很差，不足以直接部署。
- Detect-AnyShadow/LSTN 路线有价值，但不是开箱即完美。
- 用户看到过一个坏 overlay：黄色轮廓把左上角黑色物体、纸面部分亮区域等误当成阴影，用户非常不满意。
- 主要问题不是“第一帧没有阴影”，而是 seed/传播/筛选策略不稳定，且没有强约束“只追踪纸面阴影”。

问题原因推断：

1. LSTN 是通用视频目标传播。它会传播“第一帧标出的视觉区域”，不是理解“纸面阴影”这个语义。
2. 如果第一帧 seed 过大或包含非阴影，后面一定会漂移。
3. 纸面阴影是随光照变化的非刚性区域，不像物体那样有稳定纹理，传播模型容易跟丢或贴到相似暗区。
4. 手、笔、机械臂、桌面暗块和真实阴影在低分辨率下外观相近，需要额外 exclude/paper 约束。

## 9. 建议下一步

### 9.1 短期：先把 demo 用对

在 FastAPI demo 里：

- 第一帧不要涂太大。
- 只涂“确认为纸面阴影”的核心区域，不要碰到手、笔、桌面、纸外暗区。
- `max_frames` 先设 60 或 90，不要一上来全视频。
- 如果阴影明显移动，建议每隔一段重新设关键帧，而不是只依赖第一帧传播全程。

### 9.2 中期：多关键帧 teacher

当前目标是“第一帧画出阴影识别之后阴影”，但真实视频里阴影会形变/移动，只靠第一帧可能不够。更稳的路线是：

- 第 1 帧、阴影明显变化帧、手/机械臂进入帧各画一次 seed。
- 每段用 LSTN 传播。
- 合并 mask。
- 再给 tiny student 训练。

### 9.3 单场景过拟合需要多少视频

如果目标是当前机械臂/纸面/灯光系统内可用：

- 最少 5-8 段视频。
- 推荐 12-20 段视频。
- 每段 10-30 秒。
- 覆盖不同光角度、纸张位置、机械臂位置、手进入/离开、阴影强弱、背景暗物体。

不需要每帧人工标注。建议每段抽 10-20 张关键帧做修正，剩下用 teacher 传播。

## 10. ESP32-P4 当前代码状态

### 10.1 原始可显示底座

用户提供过原始屏幕工程：

```text
work/vendor/11_video_lcd_display
```

这版是用户确认能在屏幕实时显示相机内容的底座。后续 P4 部署应该以这版为唯一底座，不要再用之前黑屏的管线。

### 10.2 旧 P4 overlay 工程

当前仓库有：

```text
esp32p4_shadow_overlay/
```

这个工程里有：

- 相机/LCD overlay 尝试。
- `shadow_ai` 推理相关代码。
- PWM 舵机键盘控制代码。

但它不是当前“最稳屏幕版本”。之前黑屏问题说明不能直接把整个旧工程当底座。

### 10.3 PWM 舵机代码

PWM 代码位置：

```text
esp32p4_shadow_overlay/main/servo_keyboard.c
esp32p4_shadow_overlay/main/servo_keyboard.h
```

配置位置：

```text
esp32p4_shadow_overlay/main/Kconfig.projbuild
```

默认 3 路 GPIO：

```text
servo1 = GPIO1
servo2 = GPIO2
servo3 = GPIO3
```

键盘：

```text
q/a = 舵机1 +2/-2度
w/s = 舵机2 +2/-2度
e/d = 舵机3 +2/-2度

Q/A = 舵机1 +10/-10度
W/S = 舵机2 +10/-10度
E/D = 舵机3 +10/-10度

r = 三个舵机回中
p = 打印角度
h = 帮助
```

如果要把 PWM 加回稳定屏幕版本，只迁移 `servo_keyboard.c/.h`、Kconfig 配置和 CMake 依赖，不要迁移旧主循环。

## 11. 训练路线历史总结

我们尝试过/规划过这些阶段：

1. 用 ISTD 全量训练 tiny shadow segmentation。
2. 做 synthetic hard negatives，加入手/物体暗区 exclude。
3. 训练 modular/two-head/three-head tiny student。
4. 接入 ShadowTransfer，扩大公开阴影数据。
5. 计划接入 SD7K、EgoHOS/EgoHands、COCO/LVIS 作为更强数据源。
6. 发现纯公开数据泛化到用户机械臂/纸面场景仍差。
7. 转向真实视频微调和 teacher 伪标注。
8. 用户提出 Detect-AnyShadow，认为效果看起来可能更好。
9. 当前正在跑通 AnyShadow 第一帧交互式 seed + 后续传播 demo。

核心判断：

- 直接把 SAM/YOLO/Transformer 上 ESP32-P4 不现实。
- 它们适合做电脑端 teacher。
- P4 上最终仍应跑小 student 模型或轻量后处理。
- 目前 teacher 质量还不够，所以不要急着部署 P4。

## 12. 新窗口接手建议命令

### 12.1 查看当前服务

```bash
cd /Users/kongfans/Documents/Codex/2026-06-01/ai-ai-esp32-hugging-face-plugin
ps aux | rg "detect_anyshadow_fastapi_demo|7871"
```

### 12.2 启动 AnyShadow demo

```bash
cd /Users/kongfans/Documents/Codex/2026-06-01/ai-ai-esp32-hugging-face-plugin
work/.venv_shadow/bin/python scripts/detect_anyshadow_fastapi_demo.py
```

浏览器打开：

```text
http://127.0.0.1:7871
```

### 12.3 查看输出

```bash
find work/video_shadow_compare/anyshadow_fastapi_demo -maxdepth 3 -type f -name 'overlay.mp4' -print
open work/video_shadow_compare/anyshadow_fastapi_demo/<run_id>/overlay.mp4
```

### 12.4 快速查看 Git 改动

```bash
git status --short
```

注意：当前仓库有大量未提交修改和未跟踪文件，不要随手 reset 或 checkout。

## 13. 接手时最重要的判断

不要继续盲目训练 tiny ESP32 模型，也不要直接把坏 teacher 输出喂给 student。

下一步应该先把 teacher 做到“用户肉眼看起来至少方向对”：

1. 让用户在 FastAPI demo 里手动画第一帧阴影。
2. 先跑 60-90 帧。
3. 如果漂移，增加多关键帧 seed。
4. 如果把手/机械臂当阴影，加入 paper ROI 和 exclude mask 约束。
5. 只有 teacher 输出基本可信后，再做真实视频微调和 P4 student 蒸馏。

