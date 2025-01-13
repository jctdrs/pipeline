from typing import List
from typing import Optional

from setup.geometry_validation import Geometry
from setup.bands_validation import Band

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
    def pass_body_to_geometry(self):
        if "geometry" not in self:
            self["geometry"] = Geometry(body=self["body"])
        else:
            self["geometry"]["body"] = self["body"]
        return self
