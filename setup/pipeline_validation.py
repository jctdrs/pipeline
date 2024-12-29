from typing_extensions import Optional
from typing_extensions import List

from setup import config_validation
from setup import data_validation
from setup import parameters_validation

from pydantic import BaseModel
from pydantic import model_validator


class PipelineStep(BaseModel):
    step: str
    parameters: BaseModel
    diagnosis: Optional[bool] = False

    # All possible pipeline steps have their own validator. One way of creating
    # that would be to use factory method design.
    @model_validator(mode="before")
    def pass_step_to_parameters(self):
        self["parameters"] = parameters_validation.factory_method(
            self["step"], **self["parameters"]
        )
        return self

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
        }

        if self.step not in hip_possible_steps:
            msg = f"[ERROR] Step '{self.step}' not available."
            raise ValueError(msg)
        return self


class Meta(BaseModel):
    # Individual fields in 'Meta' are not mandatory. In case they are not
    # defined by the user, then they are manually filled with default values.
    name: str = "Default"
    author: str = "Default"
    version: str = "Default"
    description: str = "Default"


class Pipeline(BaseModel):
    # The whole 'meta' section is not mandatory to define. In case it is not
    # defined by the user, then it is manually filled with all default values.
    meta: Optional[Meta] = Meta(
        name="Default",
        author="Default",
        version="Default",
        description="Default",
    )
    config: config_validation.Config
    data: data_validation.Data
    pipeline: List[PipelineStep]
