import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring
import copy
import yaml
import cv2
import onnx
import time
import numpy as np
import colorsys
from pathlib import Path
import logging, coloredlogs


def load_yaml(yaml_path: str) -> dict:
    """通过id找到名称

    Args:
        yaml_path (str): yaml文件路径

    Returns:
        yaml (dict)
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)

    return y


def get_image(image_path: str) -> np.ndarray:
    """获取图像

    Args:
        image_path (str): 图片路径

    Returns:
        Tuple: 原图
    """
    image_bgr = cv2.imread(str(Path(image_path)))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # BGR2RGB
    return image_rgb


def resize_and_pad(image: np.ndarray, new_shape: tuple[int]) -> tuple[np.ndarray, float]:
    """缩放图片并填充为正方形,右下填充

    Args:
        image (np.Array):      图片
        new_shape (list[int]): [h, w]

    Returns:
        Tuple: 缩放的图片, 缩放比例
    """
    old_size = image.shape[:2]
    ratio = min(new_shape[0] / old_size[0], new_shape[1] / old_size[1])
    new_size = [int(x * ratio) for x in old_size]
    # 缩放高宽的长边为640
    image = cv2.resize(image, (new_size[1], new_size[0]))
    # 填充bottom和right的长度
    delta_h = new_shape[0] - new_size[0]
    delta_w = new_shape[1] - new_size[1]
    # 使用灰色填充到640*640的形状
    color = [128, 128, 128]
    # 右下方向添加灰条
    image_reized = cv2.copyMakeBorder(
        src=image,
        top=0,
        bottom=delta_h,
        left=0,
        right=delta_w,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    return image_reized, ratio


def transform(image: np.ndarray, openvino_preprocess: bool = False) -> np.ndarray:
    """图片预处理

    Args:
        image (np.ndarray): 经过缩放的图片
        openvino_preprocess (bool, optional): 是否使用了openvino的图片预处理. Defaults to False.

    Returns:
        np.ndarray: 经过预处理的图片
    """
    image = image.transpose(2, 0, 1).astype(np.float32)        # [H, W, C] -> [C, H, W]

    # openvino预处理会自动处理scale
    if not openvino_preprocess:
        image /= 255.0                      # 归一化

    return np.expand_dims(image, 0)  # [C, H, W] -> [B, C, H, W]


def mulit_colors(num_classes: int):
    #---------------------------------------------------#
    #   https://github.com/bubbliiiing/yolov8-pytorch/blob/master/yolo.py#L88
    #   画框设置不同的颜色
    #---------------------------------------------------#
    #              hue saturation value
    hsv_tuples = [(x / num_classes, 0.7, 1.) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    colors = [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in colors]
    return colors


def check_onnx(onnx_path, logger: logging.Logger):
    """检查onnx模型是否损坏

    Args:
        onnx_path (str): onnx模型路径
    """
    # 载入onnx模块
    model_ = onnx.load(onnx_path)
    # print(model_)
    # 检查IR是否良好
    try:
        onnx.checker.check_model(model_)
    except Exception:
        logger.error("Model incorrect !")
    else:
        logger.info("Model correct !")


def np_softmax(array: np.ndarray, axis=-1) -> np.ndarray:
    array -= np.max(array)
    array = np.exp(array)
    return array / np.sum(array, axis=axis)


def ignore_box2_or_not(box1: list, box2: list, ratio: float = 0.75) -> bool:
    """determine whether ignore box2 use iou

    Args:
        box1 (list): 假设外部盒子 [x_min, y_min, x_max, y_max]
        box2 (list): 假设内部盒子 [x_min, y_min, x_max, y_max]
        ratio (float): inner_box相当于box2的面积的阈值,大于阈值就忽略. Defaults to 0.75.

    Returns:
        bool: 外部盒子是否包含内部盒子
    """
    # 内部盒子面积
    inner_box_x1 = max(box1[0], box2[0])
    inner_box_y1 = max(box1[1], box2[1])
    inner_box_x2 = min(box1[2], box2[2])
    inner_box_y2 = min(box1[3], box2[3])
    # max 用来判断是否重叠
    inner_box_area = max(inner_box_x2 - inner_box_x1, 0) * max(inner_box_y2 - inner_box_y1, 0)

    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if inner_box_area / box2_area > ratio:
        return True
    else:
        return False


def ignore_overlap_boxes(detections: np.ndarray) -> np.ndarray:
    """忽略一些框,根据同一个类别的框是否包含另一个框

    Args:
        detections (np.ndarray): np.float32
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]

    Returns:
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]
    """
    # 只有1个框或者没有框就返回
    if len(detections) <= 1:
        return detections

    new_detections = []

    # 获取每个类别
    classes = np.unique(detections[:, 0])
    # 遍历单一类别
    for cls in classes:
        dets_sig_cls = detections[detections[:, 0] == cls]
        # 如果一个类别只有1个框,就直接保存
        if len(dets_sig_cls) == 1:
            new_detections.append(dets_sig_cls)
            continue
        # 求面积,根据面积排序,不是最好的办法
        h = dets_sig_cls[:, 5] - dets_sig_cls[:, 3]
        w = dets_sig_cls[:, 4] - dets_sig_cls[:, 2]
        area = np.array(h * w)
        index = area.argsort()              # 得到面积排序index
        index = index[::-1]                 # 转换为降序

        # max_i代表大的框index,min_i代表小的框index,所以不是顺序的,会出现类似 [3,0,1,4,2]的顺序,保存时也保存对应的位置上,对应原数据
        keeps = []
        for i, max_i in enumerate(index[:-1]):
            # 默认都不包含
            keep = [False] * len(dets_sig_cls)
            for min_i in index[i+1:]:
                isin = ignore_box2_or_not(dets_sig_cls[max_i, 2:], dets_sig_cls[min_i, 2:])
                keep[min_i] = isin
            keeps.append(keep)
        # 取反,原本False为不包含,True为包含,取反后False为不保留,True为保留
        keeps = ~np.array(keeps)
        # print(keeps)
        # 每一行代表被判断的框相对于判断框是否要保留
        # 每一列代表对应index的框是否保留
        # [[True, True, True, True, False, True,  True,  True, True,  True,  True,  False],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  False, True, True,  True,  False, True],
        #  [True, True, True, True, True,  False, True,  True, False, False, True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  False, True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True]]

        # 最终保留的index,True/False
        # keeps.T: 转置之后每行代表是否要保留这个框
        final_keep = np.all(keeps.T, axis=-1)
        new_detections.append(dets_sig_cls[final_keep])

    # new_detections：[np.ndarray, np.ndarray...]
    return np.concatenate(new_detections, axis=0)


def indent(elem, level=0):
    """缩进xml
    https://www.cnblogs.com/muffled/p/3462157.html
    """
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


xml_string = """
<annotation>
	<folder>VOC2007</folder>
	<filename>000001.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<owner>
		<flickrid>Fried Camels</flickrid>
		<name>Jinky the Fruit Bat</name>
	</owner>
	<size>
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>dog</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
</annotation>
"""
root = fromstring(xml_string)
# 获取临时object
base_object = copy.deepcopy(root.find("object"))


def array2xml(data: np.ndarray, shape: tuple[int], names: list[str], path: str, file_name: str):
    """将检测的array转换为xml并保存

    Args:
        data (np.ndarray):  array数据
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]
        shape (tuple[int]): 原图shape. (h, w, c)
        names (list[str]):  名称列表
        path (str):         保存路径
        file_name (str):    文件名
    """
    # 删除全部的object
    for o in root.findall("object"):
        root.remove(o)

    # 保存文件名
    root.find("filename").text = file_name + ".jpg"

    # 保存图片大小通道
    root.find("size").find('height').text = str(shape[0])
    root.find("size").find('width').text  = str(shape[1])
    root.find("size").find('depth').text  = str(shape[2])

    # 循环遍历保存框
    for rectange in data:
        # 需要重新copy,不然多个框只会保存最后一个
        temp_object = copy.deepcopy(base_object)
        # 保存类别名称和坐标
        temp_object.find("name").text = names[int(rectange[0])]

        temp_object.find("bndbox").find("xmin").text = str(int(rectange[2]))
        temp_object.find("bndbox").find("ymin").text = str(int(rectange[3]))
        temp_object.find("bndbox").find("xmax").text = str(int(rectange[4]))
        temp_object.find("bndbox").find("ymax").text = str(int(rectange[5]))

        # 将框保存起来
        root.append(temp_object)

    # 缩进root
    indent(root)
    new_tree = ET.ElementTree(root)
    xml_path = Path(path) / (file_name + ".xml")
    # 打开使用utf-8,写入时也需要utf-8
    new_tree.write(xml_path, encoding="utf-8")


def json2xml(data: dict, path: str, file_name: str):
    """将检测的json转换为xml并保存

    Args:
        data (dict):     json数据
        path (str):      保存路径
        file_name (str): 文件名
    """
    # 删除全部的object
    for o in root.findall("object"):
        root.remove(o)

    # 保存文件名
    root.find("filename").text = file_name + ".jpg"

    # 保存图片大小通道
    root.find("size").find('height').text = str(data["image_size"][0])
    root.find("size").find('width').text  = str(data["image_size"][1])
    root.find("size").find('depth').text  = str(data["image_size"][2])

    # 循环遍历保存框
    rectangles = data["detect"]
    for rectange in rectangles:
        # 需要重新copy,不然多个框只会保存最后一个
        temp_object = copy.deepcopy(base_object)
        # 保存类别名称和坐标
        temp_object.find("name").text = rectange["class"]

        temp_object.find("bndbox").find("xmin").text = str(int(rectange["box"][0]))
        temp_object.find("bndbox").find("ymin").text = str(int(rectange["box"][1]))
        temp_object.find("bndbox").find("xmax").text = str(int(rectange["box"][2]))
        temp_object.find("bndbox").find("ymax").text = str(int(rectange["box"][3]))

        # 将框保存起来
        root.append(temp_object)

    # 缩进root
    indent(root)
    new_tree = ET.ElementTree(root)
    xml_path = Path(path) / (file_name + ".xml")
    # 打开使用utf-8,写入时也需要utf-8
    new_tree.write(xml_path, encoding="utf-8")


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """将xyhw格式的坐标转换为xyxy格式的坐标
        xyhw指的是 x_center, y_center, w, h

    Args:
        x (np.ndarray): x_center, y_center, w, h 形状的数据

    Returns:
        np.ndarray: xmin, ymin, xmax, ymax 形状的数据
    """
    y = x.copy()
    y[..., 0] -= y[..., 2] / 2  # x_center -> xmin
    y[..., 1] -= y[..., 3] / 2  # y_center -> ymin
    y[..., 2] += y[..., 0]      # w -> xmax
    y[..., 3] += y[..., 1]      # h -> ymax
    return y


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """将xyxy格式的坐标转换为xywh格式的坐标
        xyhw指的是 x_center, y_center, w, h

    Args:
        x (np.ndarray): xmin, ymin, xmax, ymax 形状的数据

    Returns:
        np.ndarray: x_center, y_center, w, h 形状的数据
    """
    w = x[..., 2] = x[..., 0]
    h = x[..., 3] = x[..., 1]
    y = x.copy()
    y[..., 0] += w / 2  # xmin -> x_center
    y[..., 1] += h / 2  # ymin -> y_center
    y[..., 2] = w       # xmax -> w
    y[..., 3] = h       # ymax -> h
    return y


def get_logger(
    save_dir: str = "./logs",
    file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
) -> logging.Logger:
    # logger
    logger: logging.Logger = logging.getLogger(name="Inference")

    # 保存log
    log_path = Path(save_dir)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    filename = f"{log_path}/{file}.log"
    logging.basicConfig(
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        filename=filename,
        level=logging.DEBUG,
        filemode="a"
    )
    # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    coloredlogs.install(level="DEBUG", logger=logger)
    return logger


if __name__ == "__main__":
    # y = load_yaml("../weights/yolov8.yaml")
    # print(y["size"])   # [640, 640]
    # print(y["stride"]) # 32
    # print(y["names"])  # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    detections = np.array([[10, 0.8, 200.04971, 196.26697, 489.98325, 424.07892],
                           [10, 0.7, 141.04881, 311.3442 , 228.94856, 408.5379 ],
                           [10, 0.6, 0.       , 303.4387 , 175.52124, 424.90558],
                           [10, 0.5, 176.42613, 0.       , 460.68604, 227.06232],
                           [10, 0.3, 384.6766 , 283.063  , 419.97977, 335.35898],
                           [10, 0.8, 97.71875 , 346.97867, 103.96518, 353.037  ],
                           [10, 0.7, 575.25476, 195.62448, 628.17926, 291.2721 ],
                           [10, 0.6, 450.49182, 1.8310547, 640.     , 292.99066],
                           [10, 0.7, 73.79396 , 368.1626 , 79.10231 , 372.40448],
                           [10, 0.9, 84.013214, 332.34296, 89.18914 , 337.10605],
                           [10, 0.8, 596.2429 , 248.21837, 601.9428 , 253.99461],
                           [10, 0.1, 372.0439 , 363.4396 , 378.0838 , 368.31393]])
    print(len(detections))      # 12
    new_detections = ignore_overlap_boxes(detections)
    print(len(new_detections))  # 5
