try:
    import onnxruntime as ort
except:
    ort = None
import numpy as np
from .inference import Inference
from .functions import check_onnx
import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY" # Enabling it can significantly reduce device memory usage


class OrtInference(Inference):
    def __init__(self, model_path: str, mode: str="cpu", **kwargs) -> None:
        """
        Args:
            model_path (str): 模型路径
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
            "cpu":  ["CPUExecutionProvider"],
            # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            "cuda": [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 8 * 1024 * 1024 * 1024, # 8GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    }),
                    "CPUExecutionProvider",
                ],
            # tensorrt
            # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
            # it is recommended you also register CUDAExecutionProvider to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.
            # set providers to ["TensorrtExecutionProvider", "CUDAExecutionProvider"] with TensorrtExecutionProvider having the higher priority.
            "tensorrt": [
                    ("TensorrtExecutionProvider", {
                        "device_id": 0,
                        "trt_max_workspace_size": 8 * 1024 * 1024 * 1024, # 8GB
                        "trt_fp16_enable": False,
                        # "trt_timing_cache_enable": True, # Enabling trt_timing_cache_enable will enable ORT TRT to use TensorRT timing cache to accelerate engine build time on a device with the same compute capability.
                    }),
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 8 * 1024 * 1024 * 1024, # 8GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    }),
                    "CPUExecutionProvider",
                ]
        }[mode]

        model = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

        # fp16输入和输出,模型是fp16格式不代表全部参数为fp16
        if model.get_inputs()[0].type[7:-1] == "float16":
            self.fp16 = True
            self.logger.info("fp16 input, fp16 model may has fp32 input")

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

    def infer(self, images: np.ndarray) -> np.ndarray:
        """推理单张图片

        fp32的onnx模型的输入输出为fp32
        fp16的onnx模型的输入输出为fp16

        Args:
            images (np.ndarray): 图片 [B, C, H, W]
        Returns:
            np.ndarray: boxes [B, 84, 8400]
        """

        # 推理
        boxes: list[np.ndarray] = self.model.run(None, {self.inputs[0].name: images})
        return boxes[0]
