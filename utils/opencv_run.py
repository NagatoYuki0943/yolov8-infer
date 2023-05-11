"""onnx图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化
"""

import cv2
import numpy as np
from .inference import Inference
from .functions import check_onnx


class OpenCVInference(Inference):
    def __init__(self, model_path: str, **kwargs) -> None:
        """
        Args:
            model_path (str): 模型路径
        """
        super().__init__(**kwargs)

        self.openvino_preprocess = False    # TODO: 更好的方式将openvino_preprocess在不使用openvino时设置为False

        # 1.检测onnx模型
        check_onnx(model_path, self.logger)
        # 2.载入模型
        self.model = cv2.dnn.readNetFromONNX(model_path)
        # 3.预热模型
        self.warm_up()

    def infer(self, images: np.ndarray) -> np.ndarray:
        """推理单张图片
        Args:
            images (np.ndarray): 图片 [B, C, H, W]
        Returns:
            np.ndarray: boxes [B, 25200, 85]
        """
        # 推理
        self.model.setInput(images)
        boxes = self.model.forward()
        return boxes
