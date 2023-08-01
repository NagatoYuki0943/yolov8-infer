# 运行yolov8导出的onnx,engine,openvino等

[ultralytics/ultralytics: NEW - YOLOv8 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/ultralytics)

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

> `onnxruntime-gpu` 使用显卡要使用 `cuda` 和 `cudnn`
>
> [NVIDIA - CUDA | onnxruntime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

```python
from utils import get_image, OrtInference
import cv2


config = {
    "model_path":           r"./weights/yolov8s.onnx",
    "mode":                 r"cuda", # tensorrt cuda cpu
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference  = OrtInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb  = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
SAVE_PATH  = r"./ort_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# OpenVINO推理例子

> 安装openvino方法请看 [openvino安装](#openvino安装)

```python
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
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
SAVE_PATH  = r"./ov_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# TensorRT推理例子

> 安装tensorrt方法请看 [tensorrt安装](#tensorrt安装)
>
> 注意yolov8导出的engine会在engine文件开始添加metadata，trtexec导出的模型不会添加，因此注意engine模型和`trtexec`参数

```python
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
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
SAVE_PATH  = r"./trt_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

## 环境变量example

### windows

```sh
# tensorrt
D:\code\TensorRT\bin
D:\code\TensorRT\lib
```

### linux

> bash&zsh

```sh
# cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# tensorrt
export PATH=/home/TensorRT/bin:$PATH
export LD_LIBRARY_PATH=/home/TensorRT/lib:$LD_LIBRARY_PATH
```

# OpenCV推理例子

```python
from utils import get_image, OpenCVInference
import cv2


config = {
    "model_path":           r"./weights/yolov8s.opset12.onnx", # 导出时必须为 opset=12 https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python
    "yaml_path":            r"./weights/yolov8.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference  = OpenCVInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb  = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
SAVE_PATH  = r"./opencv_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# [Export](https://docs.ultralytics.com/modes/export/)

Export settings for YOLO models refer to the various configurations and options used to save or export the model for use in other environments or platforms. These settings can affect the model's performance, size, and compatibility with different systems. Some common YOLO export settings include the format of the exported model file (e.g. ONNX, TensorFlow SavedModel), the device on which the model will be run (e.g. CPU, GPU), and the presence of additional features such as masks or multiple labels per box. Other factors that may affect the export process include the specific task the model is being used for and the requirements or constraints of the target environment or platform. It is important to carefully consider and configure these settings to ensure that the exported model is optimized for the intended use case and can be used effectively in the target environment.

## Arguments

| Key         | Value           | Description                                          |
| :---------- | :-------------- | :--------------------------------------------------- |
| `format`    | `'torchscript'` | format to export to                                  |
| `imgsz`     | `640`           | image size as scalar or (h, w) list, i.e. (640, 480) |
| `keras`     | `False`         | use Keras for TF SavedModel export                   |
| `optimize`  | `False`         | TorchScript: optimize for mobile                     |
| `half`      | `False`         | FP16 quantization                                    |
| `int8`      | `False`         | INT8 quantization                                    |
| `dynamic`   | `False`         | ONNX/TensorRT: dynamic axes                          |
| `simplify`  | `False`         | ONNX/TensorRT: simplify model                        |
| `opset`     | `None`          | ONNX: opset version (optional, defaults to latest)   |
| `workspace` | `4`             | TensorRT: workspace size (GB)                        |
| `nms`       | `False`         | CoreML: add NMS                                      |

## Export Formats

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`.

| Format                                                       | `format` Argument | Model                     | Metadata | Arguments                                           |
| :----------------------------------------------------------- | :---------------- | :------------------------ | :------- | :-------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                              | -                 | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)      | `torchscript`     | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                     | `onnx`            | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)       | `openvino`        | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)            | `engine`          | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)               | `coreml`          | `yolov8n.mlmodel`         | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model) | `saved_model`     | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                   | `tflite`          | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)   | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                       | `tfjs`            | `yolov8n_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)              | `paddle`          | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [NCNN](https://github.com/Tencent/ncnn)                      | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

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

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=0 half=True 	  		     # half=True only compatible with GPU export, i.e. use device=0

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=cpu dynamic=True 			 # dynamic only compatible with cpu

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=cpu half=True dynamic=True  # 导出失败 half=True not compatible with dynamic=True, i.e. use only one.
```

### opencv使用的onnx

> https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python

```sh
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=0 opset=12 				# opset必须为12

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=onnx simplify=True device=0 half=True opset=12 		# opset必须为12

# opencv不支持dynamic
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
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=ncnn simplify=True device=0 # 可以用simplify的onnx

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=ncnn simplify=True device=0 half=True
```

## ncnn

```sh
yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=engine simplify=True device=0 # 可以用simplify的onnx

yolo task=detect mode=export imgsz=640 model=weights/yolov8s.pt format=engine simplify=True device=0 half=True
```

## onnx openvino tensorrt

> 目前不支持同时导出多种格式，每种格式都要单独导出

# openvino安装

## [下载英特尔® 发行版 OpenVINO™ 工具套件 (intel.cn)](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html)

一般只需要 `pip install openvino-dev==version` 即可

## openvino数据预处理

 https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw

 https://blog.csdn.net/sandmangu/article/details/107181289

https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html

## 多种推理模式

https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html

https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html

## 通过openvino的`mo`命令将onnx转换为openvino格式(支持**fp16**)

> https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html

```sh
mo --input_model "onnx_path" --output_dir "output_path" --compress_to_fp16
```

```sh
> mo --help
usage: main.py [options]

options:
  -h, --help            show this help message and exit
  --framework {paddle,tf,mxnet,caffe,kaldi,onnx}
                        Name of the framework used to train the input model.

Framework-agnostic parameters:
  --input_model INPUT_MODEL, -w INPUT_MODEL, -m INPUT_MODEL
                        {} Tensorflow*: a file with a pre-trained model (binary or text .pb file after freezing). Caffe*: a model proto file with model weights
  --model_name MODEL_NAME, -n MODEL_NAME
                        Model_name parameter passed to the final create_ir transform. This parameter is used to name a network in a generated IR and output .xml/.bin files.
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory that stores the generated IR. By default, it is the directory from where the Model Optimizer is launched.
  --input_shape INPUT_SHAPE
                        Input shape(s) that should be fed to an input node(s) of the model. Shape is defined as a comma-separated list of integer numbers enclosed in parentheses or square brackets, for example [1,3,227,227] or (1,227,227,3), where the order of dimensions depends on the framework input layout of the model. For example, [N,C,H,W] is used for ONNX* models and [N,H,W,C] for TensorFlow* models. The shape can contain undefined dimensions (? or -1) and should fit the dimensions defined in the input operation of the graph. Boundaries of undefined dimension can be specified with ellipsis, for example [1,1..10,128,128]. One boundary can be undefined, for example [1,..100] or [1,3,1..,1..]. If there are multiple inputs in the model, --input_shape should contain definition of shape for each input separated by a comma, for example: [1,3,227,227],[2,4] for a model with two inputs with 4D and 2D shapes. Alternatively, specify shapes with the --input option.
  --scale SCALE, -s SCALE All input values coming from original network inputs will be divided by this value. When a list of inputs is overridden by the --input parameter, this scale is not applied for any input that does not match with the original input of the model.If both --mean_values and --scale are specified, the mean is subtracted first and then scale is applied regardless of the order of options in command line.
   --scale SCALE, -s SCALE  All input values coming from original network inputs will be divided by this value. When a list of inputs is overridden by the --input parameter, this scale is not applied for any input that does not match with the original input of the model.If both --mean_values and --scale are specified, the mean is subtracted first and then scale is applied regardless of the order of options in command line.
  --reverse_input_channels
                        Switch the input channels order from RGB to BGR (or vice versa). Applied to original inputs of the model if and only if a number of channels equals 3. When --mean_values/--scale_values are also specified, reversing of channels will be applied to user's input data first, so that numbers in --mean_values and --scale_values go in the order of channels used in the original model. In other words, if both options are specified, then the data flow in the model looks as following: Parameter -> ReverseInputChannels -> Mean apply-> Scale apply -> the original body of the model.
  --log_level {CRITICAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                        Logger level
  --input INPUT         Quoted list of comma-separated input nodes names with shapes, data types, and values for freezing. The order of inputs in converted model is the same as order of specified operation names. The shape and value are specified as comma-separated lists. The data type of input node is specified in braces and can have one of the values: f64 (float64), f32 (float32), f16 (float16), i64 (int64), i32 (int32), u8 (uint8), boolean (bool). Data type is optional. If it's not specified explicitly then there are two options: if input node is a parameter, data type is taken from the original node dtype, if input node is not a parameter, data type is set to f32. Example, to set `input_1` with shape [1,100], and Parameter node `sequence_len` with scalar input with value `150`, and boolean input `is_training` with `False` value use the following format: "input_1[1,100],sequence_len->150,is_training->False". Another example, use the following format to set input port 0
 of the node `node_name1` with the shape [3,4] as an input node and freeze output port 1 of the node `node_name2` with the value [20,15] of the int32 type and shape [2]: "0:node_name1[3,4],node_name2:1[2]{i32}->[20,15]".
  --output OUTPUT       The name of the output operation of the model or list of names. For TensorFlow*, do not add :0 to this name.The order of outputs in converted model is the same as order of specified operation names.
  --mean_values MEAN_VALUES, -ms MEAN_VALUES
                        Mean values to be used for the input image per channel. Values to be provided in the (R,G,B) or [R,G,B] format. Can be defined for desired input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained.
 --scale_values SCALE_VALUES
                        Scale values to be used for the input image per channel. Values are provided in the (R,G,B) or [R,G,B] format. Can be defined for desired input of the model, for example: "--scale_values data[255,255,255],info[255,255,255]". The exact meaning and order of channels depend on how the original model was trained.If both --mean_values and --scale_values are specified, the mean is subtracted first and then scale is applied regardless of the order of options in command line.
  --source_layout SOURCE_LAYOUT
                        Layout of the input or output of the model in the framework. Layout can be specified in the short form, e.g. nhwc, or in complex form, e.g. "[n,h,w,c]". Example for many names:"in_name1([n,h,w,c]),in_name2(nc),out_name1(n),out_name2(nc)".
                        Layout can be partially defined, "?" can be used to specify undefined layout for one dimension, "..." can be used to specify undefined layout for multiple dimensions, for example "?c??", "nc...", "n...c", etc.
  --target_layout TARGET_LAYOUT
                        Same as --source_layout, but specifies target layout that will be in the model after processing by ModelOptimizer.
  --layout LAYOUT       Combination of --source_layout and --target_layout. Can't be used with either of them. If model has one input it is sufficient to specify layout of this input, for example --layout nhwc. To specify layouts of many tensors, names must be provided, for example: --layout "name1(nchw),name2(nc)". It is possible to instruct ModelOptimizer to change layout, for example: --layout "name1(nhwc->nchw),name2(cn->nc)". Also "*" in long layout form can be used to fuse dimensions, for example "[n,c,...]->[n*c,...]".
  --data_type {FP16,FP32,half,float}
                        [DEPRECATED] Data type for model weights and biases. If original model has FP32 weights or biases and --data_type=FP16 is specified, FP32 model weights and biases are compressed to FP16. All intermediate data is kept in original precision.
  --compress_to_fp16 [COMPRESS_TO_FP16]
                        If the original model has FP32 weights or biases, they are compressed to FP16. All intermediate data is kept in original precision.
  --transform TRANSFORM Apply additional transformations. Usage: "--transform transformation_name1[args],transformation_name2..." where [args] is key=value pairs separated by semicolon. Examples: "--transform LowLatency2" or "--transform Pruning" or "--transform LowLatency2[use_const_initializer=False]" or "--transform "MakeStateful[param_res_names= {'input_name_1':'output_name_1','input_name_2':'output_name_2'}]"" Available transformations: "LowLatency2", "MakeStateful", "Pruning"
  --disable_fusing      [DEPRECATED] Turn off fusing of linear operations to Convolution.
  --disable_resnet_optimization
                        [DEPRECATED] Turn off ResNet optimization.
  --finegrain_fusing FINEGRAIN_FUSING
                        [DEPRECATED] Regex for layers/operations that won't be fused. Example: --finegrain_fusing Convolution1,.*Scale.*
  --enable_concat_optimization
                        [DEPRECATED] Turn on Concat optimization.
  --extensions EXTENSIONS
                        Paths or a comma-separated list of paths to libraries (.so or .dll) with extensions. For the legacy MO path (if `--use_legacy_frontend` is used), a directory or a comma-separated list of directories with extensions are supported. To disable all extensions including those that are placed at the default location, pass an empty string.
  --batch BATCH, -b BATCH
                        Input batch size
  --version             Version of Model Optimizer
  --silent SILENT       Prevent any output messages except those that correspond to log level equals ERROR, that can be set with the following option: --log_level. By default, log level is already ERROR.
  --freeze_placeholder_with_value FREEZE_PLACEHOLDER_WITH_VALUE
                        Replaces input layer with constant node with provided value, for example: "node_name->True". It will be DEPRECATED in future releases. Use --input option to specify a value for freezing.
  --static_shape        Enables IR generation for fixed input shape (folding `ShapeOf` operations and shape-calculating sub-graphs to `Constant`). Changing model input shape using the OpenVINO Runtime API in runtime may fail for such an IR.
  --disable_weights_compression
                        [DEPRECATED] Disable compression and store weights with original precision.
  --progress            Enable model conversion progress display.
  --stream_output       Switch model conversion progress display to a multiline mode.
  --transformations_config TRANSFORMATIONS_CONFIG
                        Use the configuration file with transformations description. Transformations file can be specified as relative path from the current directory, as absolute path or as arelative path from the mo root directory.
  --use_new_frontend    Force the usage of new Frontend of Model Optimizer for model conversion into IR. The new Frontend is C++ based and is available for ONNX* and PaddlePaddle* models. Model optimizer uses new Frontend for ONNX* and PaddlePaddle* by default that means `--use_new_frontend` and `--use_legacy_frontend` options are not specified.
  --use_legacy_frontend
                        Force the usage of legacy Frontend of Model Optimizer for model conversion into IR. The legacy Frontend is Python based and is available for TensorFlow*, ONNX*, MXNet*, Caffe*, and Kaldi* models.

TensorFlow*-specific parameters:
  --input_model_is_text
                        TensorFlow*: treat the input model file as a text protobuf format. If not specified, the Model Optimizer treats it as a binary file by default.
  --input_checkpoint INPUT_CHECKPOINT
                        TensorFlow*: variables file to load.
  --input_meta_graph INPUT_META_GRAPH
                        Tensorflow*: a file with a meta-graph of the model before freezing
  --saved_model_dir SAVED_MODEL_DIR
                        TensorFlow*: directory with a model in SavedModel format of TensorFlow 1.x or 2.x version.
  --saved_model_tags SAVED_MODEL_TAGS
                        Group of tag(s) of the MetaGraphDef to load, in string format, separated by ','. For tag-set contains multiple tags, all tags must be passed in.
  --tensorflow_custom_operations_config_update TENSORFLOW_CUSTOM_OPERATIONS_CONFIG_UPDATE
                        TensorFlow*: update the configuration file with node name patterns with input/output nodes information.
  --tensorflow_use_custom_operations_config TENSORFLOW_USE_CUSTOM_OPERATIONS_CONFIG
                        Use the configuration file with custom operation description.
  --tensorflow_object_detection_api_pipeline_config TENSORFLOW_OBJECT_DETECTION_API_PIPELINE_CONFIG
                        TensorFlow*: path to the pipeline configuration file used to generate model created with help of Object Detection API.
  --tensorboard_logdir TENSORBOARD_LOGDIR
                        TensorFlow*: dump the input graph to a given directory that should be used with TensorBoard.
  --tensorflow_custom_layer_libraries TENSORFLOW_CUSTOM_LAYER_LIBRARIES
                        TensorFlow*: comma separated list of shared libraries with TensorFlow* custom operations implementation.
  --disable_nhwc_to_nchw
                        [DEPRECATED] Disables the default translation from NHWC to NCHW. Since 2022.1 this option is deprecated and used only to maintain backward compatibility with previous releases.

Caffe*-specific parameters:
  --input_proto INPUT_PROTO, -d INPUT_PROTO
                        Deploy-ready prototxt file that contains a topology structure and layer attributes
  --caffe_parser_path CAFFE_PARSER_PATH
                        Path to Python Caffe* parser generated from caffe.proto
  -k K                  Path to CustomLayersMapping.xml to register custom layers
  --mean_file MEAN_FILE, -mf MEAN_FILE
                        [DEPRECATED] Mean image to be used for the input. Should be a binaryproto file
  --mean_file_offsets MEAN_FILE_OFFSETS, -mo MEAN_FILE_OFFSETS
                        [DEPRECATED] Mean image offsets to be used for the input binaryproto file. When the mean image is bigger than the expected input, it is cropped. By default, centers of the input image and the mean image are the same and the mean image is
                        cropped by dimensions of the input image. The format to pass this option is the following: "-mo (x,y)". In this case, the mean file is cropped by dimensions of the input image with offset (x,y) from the upper left corner of the mean image
  --disable_omitting_optional
                        Disable omitting optional attributes to be used for custom layers. Use this option if you want to transfer all attributes of a custom layer to IR. Default behavior is to transfer the attributes with default values and the attributes defined by the user to IR.
  --enable_flattening_nested_params
                        Enable flattening optional params to be used for custom layers. Use this option if you want to transfer attributes of a custom layer to IR with flattened nested parameters. Default behavior is to transfer the attributes without flattening nested parameters.

Mxnet-specific parameters:
  --input_symbol INPUT_SYMBOL
                        Symbol file (for example, model-symbol.json) that contains a topology structure and layer attributes
  --nd_prefix_name ND_PREFIX_NAME
                        Prefix name for args.nd and argx.nd files.
  --pretrained_model_name PRETRAINED_MODEL_NAME
                        Name of a pretrained MXNet model without extension and epoch number. This model will be merged with args.nd and argx.nd files
  --save_params_from_nd
                        Enable saving built parameters file from .nd files
  --legacy_mxnet_model  Enable MXNet loader to make a model compatible with the latest MXNet version. Use only if your model was trained with MXNet version lower than 1.0.0

Kaldi-specific parameters:
  --counts COUNTS       Path to the counts file
  --remove_output_softmax
                        Removes the SoftMax layer that is the output layer
  --remove_memory       Removes the Memory layer and use additional inputs outputs instead
```

## 代码方式

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

# tensorrt安装

1. 安装cuda https://developer.nvidia.com/cuda-toolkit
2. 安装cudnn https://developer.nvidia.cn/zh-cn/cudnn

3. 解压tensorrt https://developer.nvidia.com/zh-cn/tensorrt

4. 将 `tensort` 下的`bin`和`lib`库添加到环境变量

   ```sh
   # cuda
   export CUDA_PATH=/usr/local/cuda
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   
   # tensorrt
   export PATH=/home/TensorRT/bin:$PATH
   export LD_LIBRARY_PATH=/home/TensorRT/lib:$LD_LIBRARY_PATH
   ```

5. 安装 `tensorrt` 目录下 `python` , `onnx_graphsurgeon` 和 `graphsurgeon` 下的python包

6. 安装`pycuda` like `pip install pycuda`，安装失败可以在这个页面下载安装 `[Archived: Python Extension Packages for Windows - Christoph Gohlke (uci.edu)](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda)`

## 通过 `trtexec.exe` 导出

> 先通过yolov8导出onnx，再通过 `trtexec.exe` 导出 engine
>
> 使用 `trtexec` 导出的yolov8模型要将 `config` 中的 `trtexec` 设置为 `True`

```sh
r> trtexec --help
&&&& RUNNING TensorRT.trtexec [TensorRT v8503] # D:\code\TensorRT\bin\trtexec.exe --help
=== Model Options ===
  --uff=<file>                UFF model
  --onnx=<file>               ONNX model
  --model=<file>              Caffe model (default = no model, random weights used)
  --deploy=<file>             Caffe prototxt file
  --output=<name>[,<name>]*   Output names (it can be specified multiple times); at least one output is required for UFF and Caffe
  --uffInput=<name>,X,Y,Z     Input blob name and its dimensions (X,Y,Z=C,H,W), it can be specified multiple times; at least one is required for UFF models
  --uffNHWC                   Set if inputs are in the NHWC layout instead of NCHW (use X,Y,Z=H,W,C order in --uffInput)
=== Build Options ===
  --maxBatch                  Set max batch size and build an implicit batch engine (default = same size as --batch)
                              This option should not be used when the input model is ONNX or when dynamic shapes are provided.
  --minShapes=spec            Build with dynamic shapes using a profile with the min shapes provided
  --optShapes=spec            Build with dynamic shapes using a profile with the opt shapes provided
  --maxShapes=spec            Build with dynamic shapes using a profile with the max shapes provided
  --minShapesCalib=spec       Calibrate with dynamic shapes using a profile with the min shapes provided
  --optShapesCalib=spec       Calibrate with dynamic shapes using a profile with the opt shapes provided
  --maxShapesCalib=spec       Calibrate with dynamic shapes using a profile with the max shapes provided
                              Note: All three of min, opt and max shapes must be supplied.
                                    However, if only opt shapes is supplied then it will be expanded so
                                    that min shapes and max shapes are set to the same values as opt shapes.
                                    Input names can be wrapped with escaped single quotes (ex: \'Input:0\').
                              Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128
                              Each input shape is supplied as a key-value pair where key is the input name and
                              value is the dimensions (including the batch dimension) to be used for that input.
                              Each key-value pair has the key and value separated using a colon (:).
                              Multiple input shapes can be provided via comma-separated key-value pairs.
  --inputIOFormats=spec       Type and format of each of the input tensors (default = all inputs in fp32:chw)
                              See --outputIOFormats help for the grammar of type and format list.
                              Note: If this option is specified, please set comma-separated types and formats for all
                                    inputs following the same order as network inputs ID (even if only one input
                                    needs specifying IO format) or set the type and format once for broadcasting.
  --outputIOFormats=spec      Type and format of each of the output tensors (default = all outputs in fp32:chw)
                              Note: If this option is specified, please set comma-separated types and formats for all
                                    outputs following the same order as network outputs ID (even if only one output
                                    needs specifying IO format) or set the type and format once for broadcasting.
                              IO Formats: spec  ::= IOfmt[","spec]
                                          IOfmt ::= type:fmt
                                          type  ::= "fp32"|"fp16"|"int32"|"int8"
                                          fmt   ::= ("chw"|"chw2"|"chw4"|"hwc8"|"chw16"|"chw32"|"dhwc8"|
                                                     "cdhw32"|"hwc"|"dla_linear"|"dla_hwc4")["+"fmt]
  --workspace=N               Set workspace size in MiB.
  --memPoolSize=poolspec      Specify the size constraints of the designated memory pool(s) in MiB.
                              Note: Also accepts decimal sizes, e.g. 0.25MiB. Will be rounded down to the nearest integer bytes.
                              Pool constraint: poolspec ::= poolfmt[","poolspec]
                                               poolfmt ::= pool:sizeInMiB
                                               pool ::= "workspace"|"dlaSRAM"|"dlaLocalDRAM"|"dlaGlobalDRAM"
  --profilingVerbosity=mode   Specify profiling verbosity. mode ::= layer_names_only|detailed|none (default = layer_names_only)
  --minTiming=M               Set the minimum number of iterations used in kernel selection (default = 1)
  --avgTiming=M               Set the number of times averaged in each iteration for kernel selection (default = 8)
  --refit                     Mark the engine as refittable. This will allow the inspection of refittable layers
                              and weights within the engine.
  --sparsity=spec             Control sparsity (default = disabled).
                              Sparsity: spec ::= "disable", "enable", "force"
                              Note: Description about each of these options is as below
                                    disable = do not enable sparse tactics in the builder (this is the default)
                                    enable  = enable sparse tactics in the builder (but these tactics will only be
                                              considered if the weights have the right sparsity pattern)
                                    force   = enable sparse tactics in the builder and force-overwrite the weights to have
                                              a sparsity pattern (even if you loaded a model yourself)
  --noTF32                    Disable tf32 precision (default is to enable tf32, in addition to fp32)
  --fp16                      Enable fp16 precision, in addition to fp32 (default = disabled)
  --int8                      Enable int8 precision, in addition to fp32 (default = disabled)
  --best                      Enable all precisions to achieve the best performance (default = disabled)
  --directIO                  Avoid reformatting at network boundaries. (default = disabled)
  --precisionConstraints=spec Control precision constraint setting. (default = none)
                                  Precision Constaints: spec ::= "none" | "obey" | "prefer"
                                  none = no constraints
                                  prefer = meet precision constraints set by --layerPrecisions/--layerOutputTypes if possible
                                  obey = meet precision constraints set by --layerPrecisions/--layerOutputTypes or fail
                                         otherwise
  --layerPrecisions=spec      Control per-layer precision constraints. Effective only when precisionConstraints is set to
                              "obey" or "prefer". (default = none)
                              The specs are read left-to-right, and later ones override earlier ones. "*" can be used as a
                              layerName to specify the default precision for all the unspecified layers.
                              Per-layer precision spec ::= layerPrecision[","spec]
                                                  layerPrecision ::= layerName":"precision
                                                  precision ::= "fp32"|"fp16"|"int32"|"int8"
  --layerOutputTypes=spec     Control per-layer output type constraints. Effective only when precisionConstraints is set to
                              "obey" or "prefer". (default = none)
                              The specs are read left-to-right, and later ones override earlier ones. "*" can be used as a
                              layerName to specify the default precision for all the unspecified layers. If a layer has more than
                              one output, then multiple types separated by "+" can be provided for this layer.
                              Per-layer output type spec ::= layerOutputTypes[","spec]
                                                    layerOutputTypes ::= layerName":"type
                                                    type ::= "fp32"|"fp16"|"int32"|"int8"["+"type]
  --calib=<file>              Read INT8 calibration cache file
  --safe                      Enable build safety certified engine
  --consistency               Perform consistency checking on safety certified engine
  --restricted                Enable safety scope checking with kSAFETY_SCOPE build flag
  --saveEngine=<file>         Save the serialized engine
  --loadEngine=<file>         Load a serialized engine
  --tacticSources=tactics     Specify the tactics to be used by adding (+) or removing (-) tactics from the default
                              tactic sources (default = all available tactics).
                              Note: Currently only cuDNN, cuBLAS, cuBLAS-LT, and edge mask convolutions are listed as optional
                                    tactics.
                              Tactic Sources: tactics ::= [","tactic]
                                              tactic  ::= (+|-)lib
                                              lib     ::= "CUBLAS"|"CUBLAS_LT"|"CUDNN"|"EDGE_MASK_CONVOLUTIONS"
                                                          |"JIT_CONVOLUTIONS"
                              For example, to disable cudnn and enable cublas: --tacticSources=-CUDNN,+CUBLAS
  --noBuilderCache            Disable timing cache in builder (default is to enable timing cache)
  --heuristic                 Enable tactic selection heuristic in builder (default is to disable the heuristic)
  --timingCacheFile=<file>    Save/load the serialized global timing cache
  --preview=features          Specify preview feature to be used by adding (+) or removing (-) preview features from the default
                              Preview Features: features ::= [","feature]
                                                feature  ::= (+|-)flag
                                                flag     ::= "fasterDynamicShapes0805"
                                                             |"disableExternalTacticSourcesForCore0805"

=== Inference Options ===
  --batch=N                   Set batch size for implicit batch engines (default = 1)
                              This option should not be used when the engine is built from an ONNX model or when dynamic
                              shapes are provided when the engine is built.
  --shapes=spec               Set input shapes for dynamic shapes inference inputs.
                              Note: Input names can be wrapped with escaped single quotes (ex: \'Input:0\').
                              Example input shapes spec: input0:1x3x256x256, input1:1x3x128x128
                              Each input shape is supplied as a key-value pair where key is the input name and
                              value is the dimensions (including the batch dimension) to be used for that input.
                              Each key-value pair has the key and value separated using a colon (:).
                              Multiple input shapes can be provided via comma-separated key-value pairs.
  --loadInputs=spec           Load input values from files (default = generate random inputs). Input names can be wrapped with single quotes (ex: 'Input:0')
                              Input values spec ::= Ival[","spec]
                                           Ival ::= name":"file
  --iterations=N              Run at least N inference iterations (default = 10)
  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = 200)
  --duration=N                Run performance measurements for at least N seconds wallclock time (default = 3)
  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute (default = 0)
  --idleTime=N                Sleep N milliseconds between two continuous iterations(default = 0)
  --streams=N                 Instantiate N engines to use concurrently (default = 1)
  --exposeDMA                 Serialize DMA transfers to and from device (default = disabled).
  --noDataTransfers           Disable DMA transfers to and from device (default = enabled).
  --useManagedMemory          Use managed memory instead of separate host and device allocations (default = disabled).
  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but increase CPU usage and power (default = disabled)
  --threads                   Enable multithreading to drive engines with independent threads or speed up refitting (default = disabled)
  --useCudaGraph              Use CUDA graph to capture engine execution and then launch inference (default = disabled).
                              This flag may be ignored if the graph capture fails.
  --timeDeserialize           Time the amount of time it takes to deserialize the network and exit.
  --timeRefit                 Time the amount of time it takes to refit the engine before inference.
  --separateProfileRun        Do not attach the profiler in the benchmark run; if profiling is enabled, a second profile run will be executed (default = disabled)
  --buildOnly                 Exit after the engine has been built and skip inference perf measurement (default = disabled)
  --persistentCacheRatio      Set the persistentCacheLimit in ratio, 0.5 represent half of max persistent L2 size (default = 0)

=== Build and Inference Batch Options ===
                              When using implicit batch, the max batch size of the engine, if not given,
                              is set to the inference batch size;
                              when using explicit batch, if shapes are specified only for inference, they
                              will be used also as min/opt/max in the build profile; if shapes are
                              specified only for the build, the opt shapes will be used also for inference;
                              if both are specified, they must be compatible; and if explicit batch is
                              enabled but neither is specified, the model must provide complete static
                              dimensions, including batch size, for all inputs
                              Using ONNX models automatically forces explicit batch.

=== Reporting Options ===
  --verbose                   Use verbose logging (default = false)
  --avgRuns=N                 Report performance measurements averaged over N consecutive iterations (default = 10)
  --percentile=P1,P2,P3,...   Report performance for the P1,P2,P3,... percentages (0<=P_i<=100, 0 representing max perf, and 100 representing min perf; (default = 90,95,99%)
  --dumpRefit                 Print the refittable layers and weights from a refittable engine
  --dumpOutput                Print the output tensor(s) of the last inference iteration (default = disabled)
  --dumpProfile               Print profile information per layer (default = disabled)
  --dumpLayerInfo             Print layer information of the engine to console (default = disabled)
  --exportTimes=<file>        Write the timing results in a json file (default = disabled)
  --exportOutput=<file>       Write the output tensors to a json file (default = disabled)
  --exportProfile=<file>      Write the profile information per layer in a json file (default = disabled)
  --exportLayerInfo=<file>    Write the layer information of the engine in a json file (default = disabled)

=== System Options ===
  --device=N                  Select cuda device N (default = 0)
  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)
  --allowGPUFallback          When DLA is enabled, allow GPU fallback for unsupported layers (default = disabled)
  --plugins                   Plugin library (.so) to load (can be specified multiple times)

=== Help ===
  --help, -h                  Print this message
```

## trtexec example

```sh
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine     	        # Precision: FP32
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16	        # Precision: FP32+FP16
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --int8	        # Precision: FP32+INT8
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16 --int8	# Precision: FP32+FP16+INT8
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --best	        # Precision: FP32+FP16+INT8
```

# 参考

https://github.com/dacquaviva/yolov5-openvino-cpp-python

https://mmdeploy.readthedocs.io/zh_CN/latest/tutorial/06_introduction_to_tensorrt.html
