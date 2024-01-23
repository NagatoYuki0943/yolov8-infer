from collections import Counter
import re


def remap(data: dict, remap_dict: dict) -> dict:
    """将预测id转换为数据库id,删除不需要的数据

    Args:
        data (dict):       预测数据
        remap_dict (dict): origin id to database id dict

    Returns:
        dict:              remap的数据
    """
    new_count = []
    new_box   = []
    for box in data["detect"]:
        # 去除名字
        box.pop("class")
        # 映射id
        box["class_index"] = remap_dict[box["class_index"]]

        # 忽略new_id为0的类别,这个类别不要
        if box["class_index"] == 0:
            # 不能直接remove,会导致训练数据长度变化,后面一个的数据读取不到
            continue

        new_box.append(box)
        new_count.append(box["class_index"])

    # reamp id 计数
    data["detect"] = new_box
    data["count"] = dict(Counter(new_count))

    return data


def reformat(data: dict) -> dict:
    """修改数据格式

    Args:
        data (dict):    {
                            "detect": [
                                {
                                    "class_index": 18,
                                    "confidence": 0.9458613395690918,
                                    "box": []
                                },
                                {
                                    "class_index": 18,
                                    "confidence": 0.936194896697998,
                                    "box": []
                                },
                                {
                                    "class_index": 11,
                                    "confidence": 0.9316917061805725,
                                    "box": []
                                }
                            ],
                            "count": {
                                18: 2,
                                11: 1
                            },
                            "image_size": [
                                2000,
                                3008,
                                3
                            ]
                        }

    Returns:
        dict:           {
                            "detect": [
                                {
                                    "class_index": [
                                        18
                                    ],
                                    "confidence": [
                                        0.9458613395690918
                                    ],
                                    "box": [
                                        [],
                                        []
                                    ]
                                },
                                {
                                    "class_index": [
                                        11
                                    ],
                                    "confidence": [
                                        0.9316917061805725
                                    ],
                                    "box": [
                                        []
                                    ]
                                }
                            ],
                            "count": {
                                18: 2,
                                11: 1
                            },
                            "image_size": [
                                2000,
                                3008,
                                3
                            ]
                        }

    """
    new_data = {}
    detect = []
    for k in data["count"].keys():                      # 循环count的key
        class_ids  = []
        confidence = []
        class_dets = []
        for det in data["detect"]:                      # 循环所有框
            if k == det["class_index"]:                 # 通过count的k和框的id匹配,确定同一个类别
                class_ids.append(det["class_index"])
                confidence.append(det["confidence"])
                class_dets.append(det["box"])

        class_ids = list(set(class_ids))                # 去重id
        confidence = [max(confidence)]                  # 找最大得分

        detect.append({"class_index": class_ids, "confidence": confidence, "box": class_dets})

    new_data["detect"]     = detect
    new_data["count"]      = data["count"]
    new_data["image_size"] = data["image_size"]

    return new_data


check = re.compile(
    r'^(?:http|ftp)s?://'   # http:// or https:// or ftp:// or ftps://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
    r'localhost|'           # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?'            # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def check_is_url(url: str) -> bool:
    """检查字符串是否为url"""
    res = check.match(url)
    if res is None:
        return False
    else:
        return True