from utils import get_image, TensorRTInfer
import cv2


config = {
    "model_path":           r"./weights/yolov8s.engine",
    "trtexec":              False,  # 是否使用trtexec手动导出的engine模型,yolov8的导出会在开始添加metadata,trtexec不会添加
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference  = TensorRTInfer(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb  = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_result=False, ignore_overlap_box=False)
print(result)
SAVE_PATH  = r"./trt_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
# avg transform time: 3.546875 ms, avg infer time: 7.7890625 ms, avg nms time: 0.65625 ms, avg figure time: 11.484375 ms
