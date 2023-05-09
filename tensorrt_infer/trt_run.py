# refer: https://github.com/NagatoYuki0943/14_tensorrt-python-samples/blob/main/python/efficientdet/infer.py

import sys
sys.path.append("../")

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import json
# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

from utils import Inference, get_image


class TensorRTInfer(Inference):
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        :param model_path (str): The path to the serialized engine to load from disk.
        """
        super().__init__(**kwargs)

        self.openvino_preprocess = False    # TODO: 更好的方式将openvino_preprocess在不使用openvino时设置为False

        # Load TRT engine
        self.trtlogger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.trtlogger, namespace="")
        with open(model_path, "rb") as f, trt.Runtime(self.trtlogger) as runtime:
            assert runtime
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length  yolov8 different from yolov5 in two lines https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py#L162
            metadata = json.loads(f.read(meta_len).decode('utf-8'))   # read metadata
            self.engine = runtime.deserialize_cuda_engine(f.read())   # read engine
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []        # inputs binding
        self.outputs = []       # outputs binding
        self.allocations = []   # 分配显存空间
        for i in range(self.engine.num_bindings):
            is_input = False
            # if self.engine.binding_is_input(i):
            #     is_input = True
            # name = self.engine.get_binding_name(i)
            # dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            # shape = self.context.get_binding_shape(i)

            # trt 8.5
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.engine.get_tensor_shape(name)

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)                               # 分配显存
            host_allocation = None if is_input else np.zeros(shape, dtype)  # 分配内存
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

            self.logger.info("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))
            # Input 'images' with shape [1, 3, 640, 640] and dtype float32
            # Output 'output0' with shape [1, 25200, 85] and dtype float32

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

        # 半精度推理
        if self.inputs[0]["dtype"] == np.float16:
            self.fp16 = True

        # warm up model
        self.warm_up()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return numpy arrays.
        """
        batch = np.ascontiguousarray(batch) # 将图片内存变得连续
        # Copy I/O and Execute
        cuda.memcpy_htod(self.inputs[0]['allocation'], batch) # 将内存中的图片移动到显存上
        self.context.execute_v2(self.allocations)             # infer
        for o in range(len(self.outputs)):                    # 将显存中的结果移动到内存上
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])

        # result = [o['host_allocation'] for o in self.outputs] # 取出结果
        result = self.outputs[0]['host_allocation']

        return result


if __name__ == "__main__":
    config = {
        "model_path":           r"../weights/yolov8s.engine",
        "yaml_path":            r"../weights/yolov8.yaml",
        "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
        "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
        "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
    }

    # 实例化推理器
    inference = TensorRTInfer(**config)

    # 单张图片推理
    IMAGE_PATH   = r"../images/bus.jpg"
    SAVE_PATH    = r"./trt_det.jpg"
    image_rgb = get_image(IMAGE_PATH)
    result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
    print(result)
    cv2.imwrite(SAVE_PATH, image_bgr_detect)

    # 多张图片推理
    IMAGE_DIR = r"../../datasets/coco128/images/train2017"
    SAVE_DIR  = r"../../datasets/coco128/images/train2017_res"
    # inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True)
    # avg transform time: 3.953125 ms, avg infer time: 8.1796875 ms, avg nms time: 0.765625 ms, avg figure time: 13.390625 ms
