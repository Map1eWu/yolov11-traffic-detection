# F:\python_envs\base_\Scripts\activate.bat

import os
from ultralytics import YOLO
import argparse

def argument_parser():
    parser = argparse.ArgumentParser()
    
    # Name
    parser.add_argument("--name",           type=str,   default='person_car_detection')

    # Dir
    parser.add_argument("--weights_dir",    type=str,   default="weights/best.pt")
    parser.add_argument("--source",         type=str,   default="test_img/test.jpg")
    parser.add_argument("--data_config",    type=str,   default='coco.yaml')
    parser.add_argument("--save_dir",       type=str,   default='../weights')

    # Training
    parser.add_argument("--conf",           type=float, default=0.1)
    parser.add_argument("--iou",            type=float, default=0.5)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch",          type=int,   default=8)
    parser.add_argument("--device",         type=str,   default='cuda:0')
    parser.add_argument("--classes",        type=int,   nargs='+', default=[0, 2, 5, 7])
    parser.add_argument("--imgsz",          type=int,   default=512)
    parser.add_argument("--save_period",    type=int,   default=1)

    args = parser.parse_args()
    return args

def main():

    args = argument_parser()

    # 2. 设置工作目录
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)

    # 3. 下载COCO数据集（如果还没有）

    # 4. 加载预训练模型
    model = YOLO(args.weights_dir)  # 使用YOLOv11 small模型

    data_config = args.data_config  # COCO数据集配置文件路径

    # 5. 训练模型
    results = model.train(
        data        =   data_config,        # COCO数据集配置文件
        epochs      =   args.epochs,        # 训练轮数
        imgsz       =   args.imgsz,         # 图像尺寸 (512x512)
        batch       =   args.batch,         # 批次大小
        device      =   args.device,        # 自动选择设备 (CPU/GPU)
        classes     =   args.classes,       # 只训练person和car类别 (COCO类别索引)
        project     =   args.save_dir,      # 权重保存根目录
        name        =   args.name,          # 实验名称
        save        =   True,               # 保存模型
        save_period =   args.save_period,   # 每1轮保存一次
        val         =   True,               # 验证
        verbose     =   True,               # 详细输出
        resume      =   True,
        # lr0         =   0.0003,           # 接续第一次训练结束时的lr
        # warmup_epochs = 0,                # 不重新warmup
        plots       =   True,               # 生成训练曲线图
    )
    best_model_dir = os.path.join(args.save_dir, args.name, 'weights', 'best.pt')
    best_model = YOLO(best_model_dir)

    test_image = args.source                # 您的测试图片路径
    if os.path.exists(test_image):
        results = best_model.predict(
            source      =   test_image,
            save        =   True,           # 保存结果
            conf        =   args.conf,      # 置信度阈值
            iou         =   args.iou,       # IoU阈值
            show_labels =   True,           # 显示标签
            show_conf   =   True,           # 显示置信度
            line_width  =   2               # 边界框线宽
        )

        # 打印检测结果
        for result in results:
            print(f"检测到 {len(result.boxes)} 个目标")
            for box in result.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                if cls == 0:
                    label = "person"
                elif cls == 2:
                    label = "car"
                elif cls == 5:
                    label = "bus"
                elif cls == 7:
                    label = "truck"
                else:
                    label = f"class_{cls}"
                print(f"  {label}: 置信度 {conf:.2f}")

        print(f"结果保存至: {results[0].save_dir}")
    else:
        print(f"测试图片 {test_image} 不存在，请提供一张包含人和车辆的图片进行测试")

if __name__ == "__main__":
    main()
