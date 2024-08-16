import requests
import json
from pydantic import BaseModel
import re

def remove_between_brackets(s):
    # 匹配第一个 '[\n' 和最后一个 '\n]' 之间的部分
    pattern = r'\[\n(.*?)\n\]'
    
    # 使用 sub() 方法替换匹配到的部分为空字符串
    modified_string = re.sub(pattern, '', s, flags=re.DOTALL)
    
    return modified_string


class StatusMessage(BaseModel):
    status: bool


def send_rpa_message(message):
    pass


def get_congcong_result(query):
    res = requests.post(url="http://36.103.239.194:3100/congcong/chat",
                        data={
                            "chat_id": "111",
                            "stream": False,
                            "content": query
                        }).json()
    return res


def change_status(status):
    with open('/data/project/PaddleOCR/assets/status.json', 'r') as file:
        data = json.load(file)
    data["status"] = status
    with open('/data/project/PaddleOCR/assets/status.json', 'w') as file:
        json.dump(data, file, indent=4)


def get_status():
    with open('/data/project/PaddleOCR/assets/status.json', 'r') as file:
        data: dict = json.load(file)
    return data.get("status", True)


if __name__ == "__main__":
    change_status(True)
    print(get_status())
    # pass
