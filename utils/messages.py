from pydantic import BaseModel
from typing import List


class Response(BaseModel):
    code: int
    result: List
    ip: str
    phoneNumber: str
    message: str
    title: str
