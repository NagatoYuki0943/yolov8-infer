"""onnx图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化
"""

import sys
sys.path.append("../")

import onnxruntime as ort
import numpy as np
import cv2
from utils import Inference, check_onnx, get_image


class OrtInference(Inference):
    def __init__(self, model_path: str, mode: str="cpu", **kwargs) -> None:
        """
        Args:
            model_path (str): 模型路径
            size (list[int]): 推理图片大小 [H, W]
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        """
        super().__init__(**kwargs)

        self.openvino_preprocess = False    # TODO: 更好的方式将openvino_preprocess在不使用openvino时设置为False

        mode = mode.lower()
        assert mode in ["cpu", "cuda", "tensorrt"], "onnxruntime only support cpu, cuda and tensorrt inference."

        # 0.show some info
        self.logger.info(f"onnxruntime version: {ort.__version__}")
        # self.logger.info(f"onnxruntime all providers: {ort.get_all_providers()}")
        self.logger.info(f"onnxruntime available providers: {ort.get_available_providers()}") # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        # self.logger.info(f"ort devices: {ort.get_device()}")                                  # GPU

        # 1.检测onnx模型
        check_onnx(model_path, self.logger)
        # 2.载入模型
        if mode in ["cuda", "tensorrt"]:
            import os
            os.environ["CUDA_MODULE_LOADING"] = "LAZY" # Enabling it can significantly reduce device memory usage
            self.logger.info(f"onnxruntime CUDA_MODULE_LOADING = LAZY")
        self.model = self.get_model(model_path, mode)
        # 3.获取模型收入输出
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        # 4.预热模型
        self.warm_up()

    def get_model(self, onnx_path: str, mode: str="cpu") -> ort.InferenceSession:
        """获取onnxruntime模型
        Args:
            onnx_path (str):      模型路径
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        Returns:
            ort.InferenceSession: 模型session
        """
        self.logger.info(f"inference with {mode} !")

        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = {
            "cpu":  ['CPUExecutionProvider'],
            # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            "cuda": [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ],
            # tensorrt
            # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
            # it is recommended you also register CUDAExecutionProvider to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.
            # set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
            "tensorrt": [
                    ('TensorrtExecutionProvider', {
                        'device_id': 0,
                        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024, # 2GB
                        'trt_fp16_enable': False,
                    }),
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ]
        }[mode]

        model = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

        # 半精度推理
        if model.get_inputs()[0].type[7:-1] == "float16":
            self.fp16 = True

        #--------------------------------#
        #   查看model中的内容
        #   get_inputs()返回对象，[0]返回名字
        #--------------------------------#
        # print("model outputs: \n", model.get_inputs())    # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x0000023BA140A770>]
        self.logger.info(model.get_inputs()[0])             # NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
        self.logger.info(model.get_inputs()[0].name)        # images
        self.logger.info(model.get_inputs()[0].type)        # tensor(float) str
        self.logger.info(model.get_inputs()[0].shape)       # [1, 3, 640, 640]

        # print("model outputs: \n", model.get_outputs())   # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x0000023BA140B5B0>]
        self.logger.info(model.get_outputs()[0])            # NodeArg(name='output', type='tensor(float)', shape=[1, 25200, 85])
        self.logger.info(model.get_outputs()[0].name)       # output0
        self.logger.info(model.get_outputs()[0].type)       # tensor(float) str
        self.logger.info(model.get_outputs()[0].shape)      # [1, 25200, 85]

        return model

    def infer(self, image: np.ndarray) -> np.ndarray:
        """推理单张图片
        Args:
            image (np.ndarray): 图片 [B, C, H, W]
        Returns:
            np.ndarray: boxes [B, 25200, 85]
        """

        # 推理
        boxes = self.model.run(None, {self.inputs[0].name: image})    # 返回值为list

        return boxes[0]


if __name__ == "__main__":
    config = {
        "model_path":           r"../weights/yolov8s.onnx",
        "mode":                 r"cuda",
        "yaml_path":            r"../weights/yolov8.yaml",
        "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
        "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
        "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
    }

    # 实例化推理器
    inference = OrtInference(**config)

    # 单张图片推理
    IMAGE_PATH = r"../images/bus.jpg"
    SAVE_PATH  = r"./ort_det.jpg"
    image_rgb = get_image(IMAGE_PATH)
    result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
    print(result)
    cv2.imwrite(SAVE_PATH, image_bgr_detect)

    # 多张图片推理
    IMAGE_DIR = r"../../datasets/coco128/images/train2017"
    SAVE_DIR  = r"../../datasets/coco128/images/train2017_res"
    # inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True)
    # avg transform time: 7.3828125 ms, avg infer time: 10.765625 ms, avg nms time: 0.640625 ms, avg figure time: 13.390625 ms
