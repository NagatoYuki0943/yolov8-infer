from abc import ABC, abstractmethod
import numpy as np
import os
from collections import Counter
import logging, coloredlogs
from .functions import *


class Inference(ABC):
    def __init__(self,
                 yaml_path: str,
                 confidence_threshold: float = 0.25,
                 score_threshold:      float = 0.2,
                 nms_threshold:        float = 0.45,
                 openvino_preprocess:   bool = False,
                 ) -> None:
        """父类推理器

        Args:
            yaml_path (str):                配置文件路径
            confidence_threshold (float):   只有得分大于置信度的预测框会被保留下来,越大越严格
            score_threshold (float):        opencv nms分类得分阈值,越大越严格
            nms_threshold (float):          非极大抑制所用到的nms_iou大小,越小越严格
            openvino_preprocess (bool, optional): openvino图片预处理，只有openvino模型可用. Defaults to False.
            fp16 (bool, optional):          半精度推理. Defaults to False.
        """
        self.config               = load_yaml(yaml_path)
        self.confidence_threshold = confidence_threshold
        self.score_threshold      = score_threshold
        self.nms_threshold        = nms_threshold
        self.openvino_preprocess  = openvino_preprocess
        self.fp16                 = False   # 默认不使用半精度,使用半精度时会自动判断,在onnxruntime_infer和tensorrt_infer初始化会自动推断,openvino不需要处理

        # 获取不同颜色
        self.colors = mulit_colors(len(self.config["names"].keys()))

        # logger
        self.logger: logging.Logger = logging.getLogger(name="Inference")

        # 保存log
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                            filename="./logs/log.txt",
                            level=logging.DEBUG,
                            filemode="a")
        coloredlogs.install(level="DEBUG")
        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        coloredlogs.install(level="DEBUG", logger=self.logger)


    @abstractmethod
    def infer(self, image: np.ndarray) -> np.ndarray:
        """推理图片

        Args:
            image (np.ndarray): 图片

        Returns:
            np.ndarray: 推理结果
        """
        raise NotImplementedError


    def warm_up(self):
        """预热模型
        """
        # [B, C, H, W]
        x = np.zeros((1, 3, *self.config["imgsz"]), dtype=np.float16 if self.fp16 else np.float32)
        self.infer(x)
        self.logger.info("warmup finish")


    def nms(self, detections: np.ndarray) -> np.ndarray:
        """非极大值抑制，没有confidence,只有box和class score

        Args:
            detections (np.ndarray): 检测到的数据 (8400, 84)

        Returns:
            (np.ndarray): np.float32
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]
        """
        # 位置坐标
        loc            = detections[:, :4]
        # 分类
        cls            = detections[:, 4:]

        # 最大分类index
        max_cls_index  = cls.argmax(axis=-1)
        # 最大分类score
        max_cls_score  = cls.max(axis=-1)

        # 位置
        boxes          = loc[max_cls_score > self.confidence_threshold]
        # 置信度
        confidences    = max_cls_score[max_cls_score > self.confidence_threshold]
        # 类别index
        class_indexes  = max_cls_index[max_cls_score > self.confidence_threshold]

        # [center_x, center_y, w, h] -> [x_min, y_min, w, h]
        boxes[:, 0:2] -= boxes[:, 2:4] / 2

        # 每个类别单独做nms
        detections = []
        unique_indexes = np.unique(class_indexes)
        for unique_index in unique_indexes:
            boxes_         = boxes[class_indexes==unique_index]
            confidences_   = confidences[class_indexes==unique_index]
            class_indexes_ = class_indexes[class_indexes==unique_index]

            # nms
            nms_indexes = cv2.dnn.NMSBoxes(boxes_, confidences_, self.score_threshold, self.nms_threshold)

            # 过滤
            detections.append(np.concatenate((np.expand_dims(class_indexes_[nms_indexes], 1), np.expand_dims(confidences_[nms_indexes], 1), boxes_[nms_indexes]), axis=-1))

        # 没有检测到返回空数组
        if len(detections) == 0:
            return []
        else:
            detections = np.concatenate(detections, axis=0)

        # [x_min, y_min, w, h] -> [x_min, y_min, x_max, y_max]
        detections[:, 4:6] += detections[:, 2:4]

        # 防止框超出图片边界, 前面判断为True/False,后面选择更改的列,不选择更改的列会将整行都改为0
        detections[detections[:, 2] < 0.0, 2] = 0.0
        detections[detections[:, 3] < 0.0, 3] = 0.0
        detections[detections[:, 4] > self.config["imgsz"][1], 4] = self.config["imgsz"][1]
        detections[detections[:, 5] > self.config["imgsz"][0], 5] = self.config["imgsz"][0]
        # [
        #   [class_index, confidences, xmin, ymin, xmax, ymax],
        #   ...
        # ]
        return detections


    def box_to_origin(self, detections: np.ndarray, delta_w: int, delta_h: int, shape: np.ndarray) -> np.ndarray:
        """将将检测结果的坐标还原到原图尺寸

        Args:
            detections (np.ndarray): np.float32
                    [
                        [class_index, confidences, xmin, ymin, xmax, ymax],
                        ...
                    ]
            delta_w (int):      填充的宽
            delta_h (int):      填充的高
            shape (np.ndarray): (h, w, c)

        Returns:
            np.ndarray: same as detections
        """
        if len(detections) == 0:
            return detections

        # 还原到原图尺寸并转化为int                                                    shape: (h, w, c)
        detections[:, 2] = detections[:, 2] / ((self.config["imgsz"][1] - delta_w) / shape[1])    # xmin
        detections[:, 3] = detections[:, 3] / ((self.config["imgsz"][0] - delta_h) / shape[0])    # ymin
        detections[:, 4] = detections[:, 4] / ((self.config["imgsz"][1] - delta_w) / shape[1])    # xmax
        detections[:, 5] = detections[:, 5] / ((self.config["imgsz"][0] - delta_h) / shape[0])    # ymax
        return detections


    def figure_boxes(self, detections: np.ndarray, image: np.ndarray) -> np.ndarray:
        """将框画到原图

        Args:
            detections (np.ndarray): np.float32
                    [
                        [class_index, confidences, xmin, ymin, xmax, ymax],
                        ...
                    ]
            image (np.ndarray): 原图

        Returns:
            np.ndarray: 绘制的图
        """
        if len(detections) == 0:
            self.logger.warning("no detection")
            # 返回原图
            return image

        # Print results and save Figure with detections
        for i, detection in enumerate(detections):
            classId     = int(detection[0])
            confidence  = detection[1]
            xmin        = int(detection[2])
            ymin        = int(detection[3])
            xmax        = int(detection[4])
            ymax        = int(detection[5])
            self.logger.info(f"Bbox {i} Class: {classId}, Confidence: {'{:.2f}'.format(confidence)}, coords: [ xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax} ]")

            # 绘制框
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), self.colors[classId], 2)
            # 直接在原图上绘制文字背景，不透明
            # image = cv2.rectangle(image, (xmin, ymin - 20), (xmax, ymax)), self.colors[classId], cv2.FILLED)

            # 文字
            label = str(self.config["names"][classId]) + " " + "{:.2f}".format(confidence)
            w, h = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]  # text width, height

            # 添加文字背景
            temp_image = np.zeros(image.shape).astype(np.uint8)
            temp_image = cv2.rectangle(temp_image, (xmin, ymin - 20 if ymin > 20 else ymin + h + 10), (xmax, ymin), self.colors[classId], cv2.FILLED)
            # 叠加原图和文字背景，文字背景是透明的
            image = cv2.addWeighted(image, 1.0, temp_image, 1.0, 1)

            # 添加文字
            image = cv2.putText(img         = image,
                                text        = label,
                                org         = (xmin, ymin - 5 if ymin > 20 else ymin + h + 5),
                                fontFace    = 0,
                                fontScale   = 0.5,
                                color       = (0, 0, 0),
                                thickness   = 1,
                                lineType    = cv2.LINE_AA,
                                )

        return image


    def get_boxes(self, detections: np.ndarray, shape: np.ndarray) -> dict:
        """返回还原到原图的框

        Args:
            detections (np.ndarray): np.float32
                    [
                        [class_index, confidences, xmin, ymin, xmax, ymax],
                        ...
                    ]
            shape (np.ndarray): (h, w, c)

        Returns:
            detect (dict):  {
                            "detect":     [{"class_index": class_index, "class": "class_name", "confidence": confidence, "box": [xmin, ymin, xmax, ymax]}...],    box为int类型
                            "num":        {0: 4, 5: 1},
                            "image_size": [height, width, channel]
                            }
        """
        if len(detections) == 0:
            self.logger.warning("no detection")
            return {"detect": [], "num": {}, "image_size": shape}

        detect = {} # 结果返回一个dict
        count = []  # 类别计数
        res = []
        for i, detection in enumerate(detections):
            count.append(int(detection[0]))   # 计数
            box = [None] * 4
            box[0] = int(detection[2])    # xmin
            box[1] = int(detection[3])    # ymin
            box[2] = int(detection[4])    # xmax
            box[3] = int(detection[5])    # ymax
            res.append({"class_index": int(detection[0]), "class": self.config["names"][int(detection[0])], "confidence": detection[1], "box": box})
            self.logger.info(f"Bbox {i} Class: {int(detection[0])}, Confidence: {'{:.2f}'.format(detection[1])}, coords: [ xmin: {box[0]}, ymin: {box[1]}, xmax: {box[2]}, ymax: {box[3]} ]")

        detect["detect"] = res
        # 类别计数
        detect["count"] = dict(Counter(count))
        # 图片形状
        detect["image_size"] = shape # 添加 (h, w, c)
        return detect


    def single(self, image_rgb: np.ndarray, only_get_boxes: bool = False, ignore_overlap_box: bool = False) -> tuple[dict, np.ndarray] | tuple[dict, None]:
        """单张图片推理

        Args:
            image_rgb (np.ndarray):              rgb图片
            only_get_boxes (bool, optional):     是否只获取boxes. Defaults to False.
            ignore_overlap_box (bool, optional): 是否忽略重叠的小框,不同于nms. Defaults to False.

        Returns:
            tuple[dict, np.ndarray] | tuple[dict, None]:    预测结果和绘制好的图片
        """

        # 1. 缩放图片,扩展的宽高
        t1 = time.time()
        image_reized, delta_w ,delta_h = resize_and_pad(image_rgb, self.config["imgsz"])
        input_array = transform(image_reized, self.openvino_preprocess)

        # 2. 推理
        t2 = time.time()
        # numpy float16 速度慢 https://stackoverflow.com/questions/56697332/float16-is-much-slower-than-float32-and-float64-in-numpy
        # 传递参数时转变类型比传递后再转换要快
        boxes = self.infer(input_array.astype(np.float16) if self.fp16 else input_array).astype(np.float32)
        # print(boxes.shape)        # (1, 84, 8400)

        # 3. NMS
        t3 = time.time()
        detections = self.nms(boxes[0].T)   # [1, 84, 8400] -> [84, 8400] -> [8400, 84] ->  -> [[class_index, confidences, xmin, ymin, xmax, ymax],]

        # 4. 将坐标还原到原图尺寸
        detections = self.box_to_origin(detections, delta_w, delta_h, image_rgb.shape)
        t4 = time.time()

        # 5. 画图或者获取json
        if ignore_overlap_box:  # 忽略重叠的小框,不同于nms
            detections = ignore_overlap_boxes(detections)
        detect = self.get_boxes(detections, image_rgb.shape) # shape: (h, w, c)
        if not only_get_boxes:
            image = self.figure_boxes(detections, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        t5 = time.time()
        self.logger.info(f"transform time: {int((t2-t1) * 1000)} ms, infer time: {int((t3-t2) * 1000)} ms, nms time: {int((t4-t3) * 1000)} ms, figure time: {int((t5-t4) * 1000)} ms")

        # 6. 返回结果
        if not only_get_boxes:
            return detect, image
        else:
            return detect, None


    def multi(self, image_dir: str, save_dir: str, save_xml: bool = False, ignore_overlap_box: bool = False) -> None:
        """单张图片推理

        Args:
            image_dir (str):                     图片文件夹路径
            save_dir (str):                      图片文件夹保存路径
            save_xml (bool, optional):           是否保存xml文件. Defaults to False.
            ignore_overlap_box (bool, optional): 是否忽略重叠的小框,不同于nms. Defaults to False.
        """
        if not os.path.exists(save_dir):
            self.logger.info(f"The save path {save_dir} does not exist, it has been created")
            os.makedirs(save_dir)

        # 1.获取文件夹中所有图片
        image_paths = os.listdir(image_dir)
        image_paths = [image for image in image_paths if image.lower().endswith(("jpg", "jepg", "bmp", "png"))]

        # 记录平均时间
        trans_times  = 0.0
        infer_times  = 0.0
        nms_times    = 0.0
        figure_times = 0.0

        # 2.遍历图片
        for image_file in image_paths:
            image_path = os.path.join(image_dir, image_file)

            # 3. 获取图片,缩放的图片,扩展的宽高
            t1 = time.time()
            image_rgb = get_image(image_path)
            image_reized, delta_w ,delta_h = resize_and_pad(image_rgb, self.config["imgsz"])
            input_array = transform(image_reized, self.openvino_preprocess)

            # 4. 推理
            t2 = time.time()
            boxes = self.infer(input_array.astype(np.float16) if self.fp16 else input_array).astype(np.float32)

            # 5. NMS
            t3 = time.time()
            detections = self.nms(boxes[0].T)   # [1, 84, 8400] -> [84, 8400] -> [8400, 84] ->  -> [[class_index, confidences, xmin, ymin, xmax, ymax],]

            # 6. 将坐标还原到原图尺寸
            detections = self.box_to_origin(detections, delta_w, delta_h, image_rgb.shape)

            t4 = time.time()
            # 7. 画图
            if ignore_overlap_box: # # 忽略重叠的小框,不同于nms
                detections = ignore_overlap_boxes(detections)
            image = self.figure_boxes(detections, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            t5 = time.time()

            # 8. 记录时间
            trans_time   = int((t2-t1) * 1000)
            infer_time   = int((t3-t2) * 1000)
            nms_time     = int((t4-t3) * 1000)
            figure_time  = int((t5-t4) * 1000)
            trans_times  += trans_time
            infer_times  += infer_time
            nms_times    += nms_time
            figure_times += figure_time
            self.logger.info(f"transform time: {trans_time} ms, infer time: {infer_time} ms, nms time: {nms_time} ms, figure time: {figure_time} ms")

            # 9.保存图片
            cv2.imwrite(os.path.join(save_dir, image_file), image)
            # 10.保存xml
            if save_xml:
                array2xml(detections, image_rgb.shape, self.config["names"], save_dir, "".join(image_file.split(".")[:-1]))

        self.logger.info(f"avg transform time: {trans_times / len(image_paths)} ms, avg infer time: {infer_times / len(image_paths)} ms, avg nms time: {nms_times / len(image_paths)} ms, avg figure time: {figure_times / len(image_paths)} ms")
