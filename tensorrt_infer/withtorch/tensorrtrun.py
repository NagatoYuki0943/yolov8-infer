"""onnx图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化
"""
import sys
sys.path.append("../../")

from pathlib import Path

from typing import Sequence

import numpy as np
import cv2
import time
import torch

from engine import TRTWrapper
from utils import resize_and_pad, nms, figure_boxes, load_yaml

CONFIDENCE_THRESHOLD = 0.25 # 只有得分大于置信度的预测框会被保留下来,越大越严格
SCORE_THRESHOLD = 0.2       # 框的得分置信度,越大越严格
NMS_THRESHOLD = 0.45        # 非极大抑制所用到的nms_iou大小,越小越严格


def get_image(image_path):
    """获取图像

    Args:
        image_path (str): 图片路径

    Returns:
        Tuple: 原图, 输入的tensor, 填充的宽, 填充的高
    """
    img = cv2.imread(str(Path(image_path)))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR2RGB

    img_reized, delta_w ,delta_h = resize_and_pad(img_rgb, (640, 640))

    img_reized = img_reized.astype(np.float32)
    img_reized /= 255.0                             # 归一化

    img_reized = img_reized.transpose(2, 0, 1)      # [H, W, C] -> [C, H, W]
    input_tensor = np.expand_dims(img_reized, 0)    # [C, H, W] -> [B, C, H, W]
    input_tensor = torch.from_numpy(input_tensor)

    return img, input_tensor, delta_w ,delta_h


def get_engine_model(engine_path, output_names: Sequence[str] = ['output0']):
    """获取模型

    Args:
        onnx_path (str): 模型路径

    Returns:
        InferenceSession: 推理模型
    """
    model = TRTWrapper(engine_path, output_names)
    return model


#--------------------------------#
#   推理
#--------------------------------#
def inference():
    ENGINE_PATH  = "../../weights/yolov8s.engine"
    IMAGE_PATH = "../../images/bus.jpg"
    YAML_PATH  = "../../weights/yolov8.yaml"

    # 1.获取图片,缩放的图片,扩展的宽高
    img, input_tensor, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 2.获取模型
    model = get_engine_model(ENGINE_PATH, ['output0'])

    # 获取label
    y = load_yaml(YAML_PATH)
    index2name = y["names"]
    start = time.time()
    # 4.输入字典: {input_name: input,...}
    # 输出字典: {output_name: output,...}
    boxes = model({"images": input_tensor.cuda()})
    print(boxes)
    print(boxes['output0'].shape)                  # [1, 25200, 85]
    detections = boxes['output0'].cpu().numpy()[0]  # [25200, 85]

    # 5.Postprocessing including NMS
    detections = nms(detections, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD)
    img = figure_boxes(detections, delta_w ,delta_h, img, index2name)
    end = time.time()
    print('time:', (end - start) * 1000)

    cv2.imwrite("./engine_det.png", img)


if __name__ == "__main__":
    inference()
