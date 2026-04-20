import os
from ultralytics import YOLO

# 配置
WEIGHTS = "F:/python_envs/weights/person_car_detection5/weights/best.pt"
IMAGE   = "test_img/test1.jpg"   # 替换为你的图片路径，也可以是文件夹路径
CONF    = 0.1          # 置信度阈值
IOU     = 0.5         # IoU 阈值

LABEL_MAP = {
    0: "person",
    2: "car",
    5: "bus",
    7: "truck",
}

def detect(image_path: str):
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return

    model = YOLO(WEIGHTS)

    results = model.predict(
        source      = image_path,
        conf        = CONF,
        iou         = IOU,
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
    detect(IMAGE)
