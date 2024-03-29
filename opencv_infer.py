from utils import read_image, OpenCVInference
import cv2


config = {
    "model_path":           r"./weights/yolov8s.opset12.onnx", # 导出时必须为 opset=12 https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.6,    # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference  = OpenCVInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb  = read_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_result=False, ignore_overlap_box=False)
print(result)
SAVE_PATH  = r"./opencv_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
# avg transform time: 3.7109375 ms, avg infer time: 138.65625 ms, avg nms time: 0.484375 ms, avg figure time: 11.953125 ms
