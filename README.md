# 运行yolov5导出的onnx,engine,openvino等

下载pt,onnx地址 [Releases · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/releases)

# 参考

https://github.com/dacquaviva/yolov5-openvino-cpp-python

# 文件

1. 需要权重，如onnx，tensorrt，openvino等

2. 需要配置文件，如下格式

   必须要有 `size` 和 `names`

```yaml
# infer size
imgsz:
  - 640 # height
  - 640 # width

# down sample stride
stride: 32

# classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

# Onnxruntime推理例子

> `onnxruntime-gpu` 使用显卡要使用 `cuda` 和 `cudnn

```python
from onnxruntime_infer import OrtInference
from utils import get_image
import cv2


config = {
    "model_path":           r"./weights/yolov8s.onnx",
    "mode":                 r"cuda",
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference = OrtInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
cv2.imshow("res", image_bgr_detect)
cv2.waitKey(0)

# 多张图片推理
IMAGE_DIR = r"../datasets/coco128/images/train2017"
SAVE_DIR  = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# OpenVINO推理例子

> 安装openvino方法请看openvino文件夹的`readme.md`

```python
from openvino_infer import OVInference
from utils import get_image
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
inference = OVInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
cv2.imshow("res", image_bgr_detect)
cv2.waitKey(0)

# 多张图片推理
IMAGE_DIR = r"../datasets/coco128/images/train2017"
SAVE_DIR  = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# TensorRT推理例子

> 安装tensorrt方法请看tensorrt文件夹的`readme.md`

```python
from tensorrt_infer import TensorRTInfer
from utils import get_image
import cv2


config = {
    "model_path":           r"./weights/yolov8s.engine",
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference = TensorRTInfer(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
cv2.imshow("res", image_bgr_detect)
cv2.waitKey(0)

# 多张图片推理
IMAGE_DIR = r"../datasets/coco128/images/train2017"
SAVE_DIR  = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# [Yolov8 Export](https://docs.ultralytics.com/modes/export/)

Export settings for YOLO models refer to the various configurations and options used to save or export the model for use in other environments or platforms. These settings can affect the model's performance, size, and compatibility with different systems. Some common YOLO export settings include the format of the exported model file (e.g. ONNX, TensorFlow SavedModel), the device on which the model will be run (e.g. CPU, GPU), and the presence of additional features such as masks or multiple labels per box. Other factors that may affect the export process include the specific task the model is being used for and the requirements or constraints of the target environment or platform. It is important to carefully consider and configure these settings to ensure that the exported model is optimized for the intended use case and can be used effectively in the target environment.

| Key         | Value           | Description                                          |
| :---------- | :-------------- | :--------------------------------------------------- |
| `format`    | `'torchscript'` | format to export to                                  |
| `imgsz`     | `640`           | image size as scalar or (h, w) list, i.e. (640, 480) |
| `keras`     | `False`         | use Keras for TF SavedModel export                   |
| `optimize`  | `False`         | TorchScript: optimize for mobile                     |
| `half`      | `False`         | FP16 quantization                                    |
| `int8`      | `False`         | INT8 quantization                                    |
| `dynamic`   | `False`         | ONNX/TF/TensorRT: dynamic axes                       |
| `simplify`  | `False`         | ONNX: simplify model                                 |
| `opset`     | `None`          | ONNX: opset version (optional, defaults to latest)   |
| `workspace` | `4`             | TensorRT: workspace size (GB)                        |
| `nms`       | `False`         | CoreML: add NMS                                      |

## Export Formats

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`.

| Format                                                       | `format` Argument | Model                     | Metadata |
| :----------------------------------------------------------- | :---------------- | :------------------------ | :------- |
| [PyTorch](https://pytorch.org/)                              | -                 | `yolov8s.pt`              | ✅        |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)      | `torchscript`     | `yolov8s.torchscript`     | ✅        |
| [ONNX](https://onnx.ai/)                                     | `onnx`            | `yolov8s.onnx`            | ✅        |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)       | `openvino`        | `yolov8s_openvino_model/` | ✅        |
| [TensorRT](https://developer.nvidia.com/tensorrt)            | `engine`          | `yolov8s.engine`          | ✅        |
| [CoreML](https://github.com/apple/coremltools)               | `coreml`          | `yolov8s.mlmodel`         | ✅        |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model) | `saved_model`     | `yolov8s_saved_model/`    | ✅        |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8s.pb`              | ❌        |
| [TF Lite](https://www.tensorflow.org/lite)                   | `tflite`          | `yolov8s.tflite`          | ✅        |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)   | `edgetpu`         | `yolov8s_edgetpu.tflite`  | ✅        |
| [TF.js](https://www.tensorflow.org/js)                       | `tfjs`            | `yolov8s_web_model/`      | ✅        |
| [PaddlePaddle](https://github.com/PaddlePaddle)              | `paddle`          | `yolov8s_paddle_model/`   | ✅        |

> example

```sh
yolo export model=yolov8s.pt format=onnx  # export official model
yolo export model=path/to/best.pt format=onnx  # export custom trained model
```

## torchscript

```sh
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=torchscript device=0 
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=torchscript device=cpu optimize=True # optimize not compatible with cuda devices, i.e. use device=cpu
```

## onnx

> 注意:
>
> `onnxruntime` 和 `onnxruntime-gpu` 不要同时安装，否则使用 `gpu` 推理时速度会很慢，如果同时安装了2个包，要全部卸载，再安装`onnxruntime-gpu` 才能使用gpu推理，否则gpu速度会很慢

```sh
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=0

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=0 half=True 	  			# half=True only compatible with GPU export, i.e. use device=0

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=cpu dynamic=True 			# dynamic only compatible with cpu

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=cpu half=True dynamic=True  # 导出失败 half=True not compatible with dynamic=True, i.e. use only one.
```

## openvino

```sh
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=openvino simplify=True device=cpu # 可以用simplify的onnx
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=openvino simplify=True device=cpu half=True # v5导出时，openvino支持half,但是要使用cpu导出onnx的half会报错,所以要使用 --device 0, openvino导出和设备无关,不受影响,主要是导出onnx的问题 v8没问题
```

### 通过openvino的`mo`命令将onnx转换为openvino格式(支持**fp16**)

> https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html

```sh
mo --input_model "onnx_path" --output_dir "output_path" --compress_to_fp16

mo --input_model "onnx_path" --output_dir "output_path" --compress_to_fp16
```

#### 代码方式

```python
from openvino.tools import mo
from openvino.runtime import serialize

onnx_path = "onnx_path"

# fp32 IR model
fp32_path = "fp32_path"
output_path = fp32_path + ".xml"
print(f"Export ONNX to OpenVINO FP32 IR to: {output_path}")
model = mo.convert_model(onnx_path)
serialize(model, output_path)

# fp16 IR model
fp16_path = "fp16_path"
output_path = fp16_path + ".xml"

print(f"Export ONNX to OpenVINO FP16 IR to: {output_path}")
model = mo.convert_model(onnx_path, compress_to_fp16=True)
serialize(model, output_path)
```

### export failure  0.9s: DLL load failed while importing ie_api

> https://blog.csdn.net/qq_26815239/article/details/123047840
>
> 如果你使用的是 Python 3.8 或更高版本，并且是在Windows系统下通过pip安装的openvino，那么该错误的解决方案如下：

1. 进入目录 `your\env\site-packages\openvino\inference_engine`
2. 打开文件 `__init__.py`
3. 26行下添加一行

```python
        if os.path.isdir(lib_path):
            # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
            if (3, 8) <= sys.version_info:
                os.add_dll_directory(os.path.abspath(lib_path))
                os.environ['PATH'] = os.path.abspath(lib_path) + ';' + os.environ['PATH']	# 添加这一行
```

## tensorrt

```sh
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=engine simplify=True device=0 # 可以用simplify的onnx

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=engine simplify=True device=0 half=True

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=engine simplify=True device=0 dynamic=True batch=16			# --dynamic model requires maximum --batch-size argument

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=engine simplify=True device=0 half=True dynamic=True batch=16 # 导出失败 half=True not compatible with dynamic=True, i.e. use only one.
```

## onnx openvino tensorrt

> 目前不支持同时导出多种格式，每种格式都要单独导出
