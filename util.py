import collections
from pydantic import BaseModel


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


class LineRequestBody(BaseModel):
    line: str


class PointRequestBody(BaseModel):
    point: str
