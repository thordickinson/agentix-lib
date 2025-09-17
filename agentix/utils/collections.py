from typing import List, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

def flatten(list_of_lists: List[List[T]]) -> List[T]:
    return [item for sublist in list_of_lists for item in sublist]

def model_dump_list(list: List[BaseModel]) -> list[dict]:
    return [item.model_dump() for item in list]

