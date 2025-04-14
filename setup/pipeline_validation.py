from typing import Optional
from typing import Literal

from setup.config_validation import Config
from setup.data_validation import Data
from setup import parameters_validation

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
        "hip.rms",
    ]
    diagnosis: bool
    parameters: BaseModel


class PipelineStep(BaseModel):
    step: str
    parameters: list[BaseModel]
    diagnosis: Optional[bool] = False

    # All possible pipeline steps have their own validator. One way of creating
    # that would be to use factory method design.
    @model_validator(mode="before")
    def pass_step_to_parameters(self):
        if "parameters" not in self:
            msg = "[ERROR] 'parameters' not defined."
            raise ValueError(msg)

        for idx, model in enumerate(self["parameters"]):
            self["parameters"][idx] = parameters_validation.factory_method(
                self["step"], **self["parameters"][idx]
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
            "hip.rms",
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
        author="Default",
        description="Default",
    )
    config: Config
    data: Data
    before: Optional[list[PipelineStep]] = []
    pipeline: list[PipelineStep]
