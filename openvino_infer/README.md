# [下载英特尔® 发行版 OpenVINO™ 工具套件 (intel.cn)](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html)

一般只需要 `pip install openvino-dev==version` 即可

# openvino数据预处理

 https://mp.weixin.qq.com/s/4lkDJC95at2tK_Zd62aJxw

 https://blog.csdn.net/sandmangu/article/details/107181289

https://docs.openvino.ai/latest/openvino_2_0_preprocessing.html


# 多种推理模式

https://docs.openvino.ai/latest/openvino_2_0_inference_pipeline.html

https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html

# 通过openvino的`mo`命令将onnx转换为openvino格式(支持**fp16**)

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

