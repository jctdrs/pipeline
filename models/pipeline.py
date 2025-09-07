from typing import Optional
from typing import Literal
from typing import List
from typing import Any

from models.config import Config
from models.data import Data
from models import parameters

from pydantic import BaseModel
from pydantic import model_validator


# When building the pipeline for each band, we need to parse the parameters
# for each band name. This is exactly as PipelineStep but with the parameters
# unrolled for each step instead of a list.
class PipelineStepUnrolled(BaseModel):
    step: Literal[
        "hip.cutout",
        "hip.skySubtract",
        "hip.foregroundMask",
        "hip.degrade",
        "hip.regrid",
        "hip.integrate",
        "hip.test",
        "core.subtract",
        "core.multiply",
    ]
    diagnosis: bool
    parameters: BaseModel


class PipelineStep(BaseModel):
    step: str
    parameters: List[BaseModel]
    diagnosis: Optional[bool] = False

    # All possible pipeline steps have their own validator. One way of creating
    # that would be to use factory method design.
    @model_validator(mode="before")
    @classmethod
    def pass_step_to_parameters(cls, data: Any) -> Any:
        if "parameters" not in data:
            msg = "[ERROR] 'parameters' not defined."
            raise ValueError(msg)

        for idx, model in enumerate(data["parameters"]):
            data["parameters"][idx] = parameters.factory_method(
                data["step"], **data["parameters"][idx]
            )
        return data

    @model_validator(mode="after")
    def validate_pipeline_steps(self):
        hip_possible_steps = {
            "hip.degrade",
            "hip.skySubtract",
            "hip.cutout",
            "hip.regrid",
            "hip.integrate",
            "hip.foregroundMask",
            "hip.test",
            "core.subtract",
            "core.multiply",
        }

        if self.step not in hip_possible_steps:
            msg = f"[ERROR] Step '{self.step}' not valid."
            raise ValueError(msg)
        return self


class Meta(BaseModel):
    # Individual fields in 'Meta' are not mandatory. In case they are not
    # defined by the user, then they are manually filled with default values.
    name: str = "Default"
    description: str = "Default"


class Pipeline(BaseModel):
    # The whole 'meta' section is not mandatory to define. In case it is not
    # defined by the user, then it is manually filled with all default values.
    meta: Optional[Meta] = Meta(
        name="Default",
        description="Default",
    )
    config: Config
    data: Data
    before: Optional[List[PipelineStep]] = []
    pipeline: List[PipelineStep]

    @model_validator(mode="after")
    def check_if_regrid_before_degrade(self):
        for idx in range(1, len(self.pipeline)):
            if (
                getattr(self.pipeline[idx], "step") == "hip.degrade"
                and getattr(self.pipeline[idx - 1], "step") == "hip.regrid"
            ):
                msg = "[WARNING] It is not advisable to regrid before degrading."
                raise ValueError(msg)
        return self
