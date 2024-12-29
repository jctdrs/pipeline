from typing import List

from setup import geometry_validation
from setup import bands_validation

from pydantic import BaseModel
from pydantic import model_validator


class Data(BaseModel):
    body: str
    geometry: geometry_validation.Geometry
    bands: List[bands_validation.Band]

    # We need to use the 'before' mode in this case because the 'body'
    # attribute in 'Geometry' is not defined by the user from the specification
    # but manually in the following method.
    @model_validator(mode="before")
    def pass_body_to_geometry(self):
        self["geometry"]["body"] = self["body"]
        return self
