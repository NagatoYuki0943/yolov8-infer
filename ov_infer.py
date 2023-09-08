from utils import get_image, OVInference
import cv2


config = {
    "model_path":           r"./weights/yolov8s_openvino_model/yolov8s.xml",
    "mode":                 r"cpu",
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
    "openvino_preprocess":  True,   # 是否使用openvino图片预处理
}

# 实例化推理器
inference  = OVInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb  = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_result=False, ignore_overlap_box=False)
print(result)
SAVE_PATH  = r"./ov_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
# avg transform time: 3.2109375 ms, avg infer time: 55.2578125 ms, avg nms time: 0.5078125 ms, avg figure time: 11.234375 ms
