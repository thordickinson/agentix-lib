import json
from pydantic import BaseModel

def to_json(object: any) -> str:
    # serialize as json is not BaseModel, else use the BaseModel serialization
    if isinstance(object, BaseModel):
        return object.model_dump_json()
    else:
        return json.dumps(object)
