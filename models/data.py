from typing import Optional
from typing import List
from typing import Any

from models.geometry import Geometry
from models.bands import Band

from pydantic import BaseModel
from pydantic import model_validator


class Data(BaseModel):
    body: str
    geometry: Optional[Geometry] = None
    bands: List[Band]

    # We need to use the 'before' mode in this case because the 'body'
    # attribute in 'Geometry' is not defined by the user from the specification
    # but manually in the following method.
    @model_validator(mode="before")
    @classmethod
    def pass_body_to_geometry(cls, data: Any) -> Any:
        if "geometry" not in data:
            data["geometry"] = Geometry(body=data["body"])
        else:
            data["geometry"]["body"] = data["body"]
        return data
