import numpy as np
import cv2
import os
import time
import grpc
import json
from concurrent import futures
import base64
import object_detect_pb2
import object_detect_pb2_grpc
import asyncio
import sys
sys.path.append("../")
from utils import Inference, OrtInference, json2xml


SERVER_HOST      = r"[::]:50051"
SERVER_SAVE_PATH = r"server"
SAVE             = True # 是否保存图片和xml
os.makedirs(SERVER_SAVE_PATH, exist_ok=True)


class Server(object_detect_pb2_grpc.YoloDetectServicer):
    def __init__(self, inference) -> None:
        super().__init__()
        self.inference: Inference = inference

    async def v8_detect(self, request: object_detect_pb2.Request,
                    context: grpc.aio.ServicerContext)-> object_detect_pb2.Response:
        """接收request,返回response
        v8_detect是proto中service YoloDetecte中的rpc v8_detect
        """
        #=====================接收图片=====================#
        try:
            # 解码图片                                  image是Request中设定的变量
            image_decode    = base64.b64decode(request.image)
            # 变成一个矩阵 单维向量
            array           = np.frombuffer(image_decode, dtype=np.uint8)
            # print("array shape:", array.shape)
            # 再解码成图片 三维图片
            image_bgr       = cv2.imdecode(array, cv2.IMREAD_COLOR)
        except:
            self.inference.logger.error("decode image fail!!!")
            fake_image      = np.array([])
            fake_image_64   = base64.b64encode(fake_image)
            fake_detect_str = json.dumps({})
            return object_detect_pb2.Response(image=fake_image_64, detect=fake_detect_str)

        #=====================预测图片=====================#
        image_rgb                = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detect, image_bgr_detect = self.inference.single(image_rgb, only_get_boxes=False) # 推理返回结果和绘制的图片

        #================保存图片和检测结果=================#
        if SAVE:
            file_name = str(time.time())
            cv2.imwrite(os.path.join(SERVER_SAVE_PATH, file_name + ".jpg"), image_bgr)
            # 保存检测结果
            json2xml(detect, SERVER_SAVE_PATH, file_name)

        #=====================编码图片=====================#
        # 返回True和编码,这里只要编码
        image_encode  = cv2.imencode(".jpg", image_bgr_detect)[1]
        # image_bytes = image_encode.tobytes()
        # image_64    = base64.b64encode(image_bytes)
        image_64      = base64.b64encode(image_encode)

        #=====================编码结果=====================#
        detect_str    = json.dumps(detect)

        #==================返回图片和结果===================#
        #                                 image和detect是Response中设定的变量
        return object_detect_pb2.Response(image=image_64, detect=detect_str)


def get_inference():
    """获取推理器"""
    # 模型配置文件
    config = {
        'model_path':           r"../weights/yolov8s.onnx",
        'mode':                 r"cuda",
        'yaml_path':            r"../weights/yolov8.yaml",
        'confidence_threshold': 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
        'score_threshold':      0.2,    # opencv nms分类得分阈值,越大越严格
        'nms_threshold':        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
    }

    # 实例化推理器
    inference = OrtInference(**config)
    inference.logger.info("load inference!")
    return inference


async def run():
    # 最大客户端连接10(max_workers=10)，这里可定义最大接收和发送大小(单位M)，默认只有4M
    # 异步要用 aio
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 100 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
                            )
    # 绑定处理器
    object_detect_pb2_grpc.add_YoloDetectServicer_to_server(Server(get_inference()), server)

    # 绑定地址
    server.add_insecure_port(SERVER_HOST)
    await server.start()
    print(f'gRPC 服务端已开启，地址为 {SERVER_HOST}...')
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(run())
