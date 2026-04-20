# 🚗 基于 YOLOv11 的行人与车辆检测


![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-purple)
![COCO](https://img.shields.io/badge/数据集-COCO2017-orange)



---

本仓库提供了基于 [YOLOv11](https://github.com/ultralytics/ultralytics) 的**行人与车辆检测**模型的完整训练与推理代码。
模型在 [COCO 2017](https://cocodataset.org) 数据集上进行微调，可检测道路场景中的四类目标——
**行人、小汽车、大巴车和卡车**，支持图片、视频文件及摄像头实时推理。

模型使用 GPU 训练 **50 个 epoch**，输入分辨率为 512×512，
在 COCO 验证集上最终达到 **mAP@0.5 = 0.689**、**mAP@0.5:0.95 = 0.502**。
本项目作为完整流程示例，涵盖数据配置、模型训练与多模态推理的全过程。

---

## 📌 检测类别

| 类别 ID | 名称 |
|---------|------|
| 0 | person（行人）|
| 2 | car（小汽车）|
| 5 | bus（大巴车）|
| 7 | truck（卡车）|

---

## 🖼️ 效果展示

<!-- 放一张检测结果的截图，直观展示很重要 -->
![demo](test_img/result.jpg)

---

## 📊 训练结果

三项损失函数在训练过程中均持续下降，模型在第 40～50 epoch 间趋于收敛。
| Epoch | Loss_box | Loss_cls | Loss_dfl | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|----------|----------|----------|-----------|--------|---------|--------------|
| 1     | 1.175    | 1.058    | 1.169    | 0.632     | 0.498  | 0.556   | 0.373        |
| 10    | 1.158    | 1.026    | 1.158    | 0.709     | 0.560  | 0.641   | 0.456        |
| 20    | 1.113    | 0.966    | 1.130    | 0.711     | 0.591  | 0.661   | 0.475        |
| 30    | 1.082    | 0.919    | 1.111    | 0.735     | 0.601  | 0.674   | 0.491        |
| 40    | 1.013    | 0.825    | 1.077    | 0.759     | 0.603  | 0.689   | 0.501        |
| **50**| **0.976**| **0.772**| **1.057**| **0.759** |**0.605**|**0.689**| **0.502**   |


---

## 📦 安装依赖

```bash
git clone https://github.com/Map1eWi/yolov11-traffic-detection.git
cd yolov11-traffic-detection
pip install -r requirements.txt
```

---

## 🔽 下载数据集

下载用于训练的COCO2017数据集，放至 `dataset/` 目录：

```bash
python download.py your_download_dir
```

---

## 🚀 快速开始

**图片检测**
```bash
python detect.py --source test_img/test.jpg --weights_dir weights/best.pt
```

**视频检测**
```bash
python detect_video.py --source test_video/test.mp4 --weights_dir weights/best.pt
```

---

## 🏃 训练

训练基于 COCO 数据集，仅使用 person/car/bus/truck 四类：

```bashÂ
python train.py --data_config coco.yaml --epochs 50 --imgsz 512 --batch 8
```

**训练配置**

| 参数 | 值 |
|------|----|
| 基础模型 | YOLOv11s |
| 训练轮数 | 50 epochs |
| 图像尺寸 | 512×512 |
| 训练设备 | NVIDIA GPU (CUDA) |
| 数据集 | COCO 2017 |

---

## 📄 License

MIT License

