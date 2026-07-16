# 单场景过拟合阴影标注工具

脚本：

```bash
scripts/overfit_shadow_annotator.py
```

用途：从视频抽帧，在浏览器里手动画纸面阴影区域，保存成后续训练可用的 mask。

标注网页额外需要：

```bash
python -m pip install fastapi uvicorn pydantic
```

## 启动

把 `VIDEO_PATH` 换成你的机械臂/纸面视频：

```bash
cd /path/to/shadow-detect

work/.venv_shadow/bin/python scripts/overfit_shadow_annotator.py \
  --video "VIDEO_PATH" \
  --review work/datasets/overfit_shadow_v1_review \
  --frames 120 \
  --port 7872
```

然后打开：

```text
http://127.0.0.1:7872
```

如果要重新抽帧覆盖旧标注，加：

```bash
--overwrite
```

## 怎么涂

- `1`：黄色，纸面阴影，要让模型学会识别的区域。
- `2`：橙色，手、机械臂、笔、暗物体，明确告诉模型这些不是阴影。
- `3`：紫色，非纸面区域，可选。
- `0`：擦除。
- `[` / `]`：缩小/放大笔刷。
- `S`：保存当前帧。
- `A/D` 或左右方向键：上一帧/下一帧，切帧前会自动保存。

建议先标 60-120 张关键帧，不用逐帧标完整视频。过拟合单一场景时，阴影变化大的位置、手/机械臂进入画面的帧最重要。

## 输出格式

标注会保存在：

```text
work/datasets/overfit_shadow_v1_review/images/
work/datasets/overfit_shadow_v1_review/masks_reviewed/
work/datasets/overfit_shadow_v1_review/previews/
work/datasets/overfit_shadow_v1_review/review_manifest.csv
```

mask 标签：

```text
0 = clean/background
1 = paper_shadow
2 = hand/object/arm_exclude
3 = non_paper_optional
```

## 转成训练集

标完后运行：

```bash
work/.venv_shadow/bin/python scripts/real_video_review_to_unified.py \
  --review work/datasets/overfit_shadow_v1_review \
  --output work/datasets/overfit_shadow_v1_unified \
  --shuffle-seed 7 \
  --merge-nonpaper-into-background
```

输出会变成：

```text
work/datasets/overfit_shadow_v1_unified/train/images
work/datasets/overfit_shadow_v1_unified/train/masks
work/datasets/overfit_shadow_v1_unified/val/images
work/datasets/overfit_shadow_v1_unified/val/masks
work/datasets/overfit_shadow_v1_unified/test/images
work/datasets/overfit_shadow_v1_unified/test/masks
```

这个目录后面就可以拿去做单场景 overfitting 微调。

教师模型训练示例：

```bash
work/.venv/bin/python scripts/train.py \
  --data work/datasets/overfit_shadow_v1_unified \
  --out work/runs/overfit_shadow_teacher \
  --epochs 250 \
  --batch-size 6 \
  --base-channels 16 \
  --learning-rate 0.001 \
  --class-weights 0.5,6.0,2.0 \
  --architecture teacher \
  --no-augment
```
