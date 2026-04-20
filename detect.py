import os
from ultralytics import YOLO

import argparse

def argument_parser():
    parser = argparse.ArgumentParser()

    # Dir
    parser.add_argument("--weights_dir",    type=str,   default="github/weights/best.pt")
    parser.add_argument("--source",         type=str,   default="E:\\1\\test.mp4")

    # Training
    parser.add_argument("--conf",           type=float, default=0.25)
    parser.add_argument("--iou",            type=float, default=0.5)

    args = parser.parse_args()
    return args


def detect():

    args = argument_parser()

    LABEL_MAP = {
        0: "person",
        2: "car",
        5: "bus",
        7: "truck",
    }

    if not os.path.exists(args.source):
        print(f"图片不存在: {args.source}")
        return

    model = YOLO(args.weights_dir)

    results = model.predict(
        source      = args.source,
        conf        = args.conf,
        iou         = args.iou,
        save        = True,        # 保存标注结果图片
        show_labels = True,
        show_conf   = True,
        line_width  = 2,
    )

    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("未检测到任何目标")
            continue

        print(f"共检测到 {len(boxes)} 个目标：")
        for box in boxes:
            cls  = int(box.cls.item())
            conf = box.conf.item()
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = LABEL_MAP.get(cls, f"class_{cls}")
            print(f"  {label}: 置信度 {conf:.2f}  位置 ({x1},{y1}) -> ({x2},{y2})")

        print(f"\n标注图片保存至: {result.save_dir}")

if __name__ == "__main__":
    detect()
