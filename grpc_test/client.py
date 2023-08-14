import numpy as np
import cv2
import os
import grpc
import base64
import json
import object_detect_pb2
import object_detect_pb2_grpc



SERVER_HOST      = r"localhost:50051"
CLIENT_SAVE_PATH = r"client"
os.makedirs(CLIENT_SAVE_PATH, exist_ok=True)


def run():
    """发送request,接收response
    """
    image_url = "../images/bus.jpg"
    # image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 本次不使用SSL，所以channel是不安全的
    with grpc.insecure_channel(SERVER_HOST) as channel:
        # 客户端实例
        stub = object_detect_pb2_grpc.YoloDetectStub(channel)

        #=================发送并接收新图片==================#
        # v8_detect是proto中service YoloDetect中的rpc v8_detect
        #                                                   image_url是Request中设定的变量
        response = stub.v8_detect(object_detect_pb2.Request(image_url=image_url))

    # 解码检测结果                      detect是Response中设定的变量
    detect       = json.loads(response.detect)
    with open(os.path.join(CLIENT_SAVE_PATH, "detect.json"), mode="w", encoding="utf-8") as f:
        json.dump(detect, f, indent=4, ensure_ascii=False) # ensure_ascii=False 保存为中文
    print(detect)
    if detect == {}:
        print("detect error!!!")
        return

    # 解码图片                                image是Response中设定的变量
    image_decode = base64.b64decode(response.image)
    # 变成一个矩阵 单维向量
    array        = np.frombuffer(image_decode, dtype=np.uint8)
    # 再解码成图片 三维图片
    image        = cv2.imdecode(array, cv2.IMREAD_COLOR)
    print(image.shape, image.dtype)
    cv2.imwrite(os.path.join(CLIENT_SAVE_PATH, "bus.jpg"), image)


if __name__ == "__main__":
    run()
