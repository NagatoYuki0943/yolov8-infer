from utils import get_image, OrtInference
import cv2


config = {
    "model_path":           r"./weights/yolov8s.onnx",
    "mode":                 r"cuda", # tensorrt cuda cpu
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.6,    # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference  = OrtInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb  = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_result=False, ignore_overlap_box=False)
print(result)
SAVE_PATH  = r"./ort_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
# tensorrt: avg transform time: 3.6328125 ms, avg infer time: 26.7109375 ms, avg nms time: 0.5234375 ms, avg figure time: 11.328125 ms
# cuda:     avg transform time: 3.546875 ms, avg infer time: 28.7578125 ms, avg nms time: 0.5234375 ms, avg figure time: 11.65625 ms
# cpu:      avg transform time: 4.5390625 ms, avg infer time: 90.7421875 ms, avg nms time: 0.78125 ms, avg figure time: 13.015625 ms
