import os
from typing import Optional
from typing import Dict

from pydantic import BaseModel
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator


# A factory method design to return the correct validator for each step of the
# pipeline depending on what the user needs. Each validator is unique to the
# nature of the step.
def factory_method(pipeline_step: str, **parameters) -> BaseModel:
    pipeline_step_class = Interface.get(pipeline_step)
    if pipeline_step_class is None:
        msg = f"[ERROR] Pipeline step {pipeline_step} not valid."
        raise ValueError(msg)
    return pipeline_step_class(**parameters)


class HIPDegrade(BaseModel):
    kernel: Optional[str] = None
    target: Optional[int] = None
    band: str

    @model_validator(mode="after")
    def check_if_kernel_path_exists(self):
        if self.kernel is not None:
            if not os.path.exists(self.kernel):
                msg = f"[ERROR] Path {self.kernel} not found."
                raise OSError(msg)
        return self

    @model_validator(mode="after")
    def check_if_kernel_and_target_are_defined(self):
        if self.kernel is not None and self.target is not None:
            msg = "[ERROR] Both 'target' and 'kernel' are defined."
            raise ValueError(msg)
        return self


class HIPSkySubtract(BaseModel):
    cellFactor: Optional[PositiveInt] = 1
    band: str


class HIPRegrid(BaseModel):
    target: str
    band: str

    @model_validator(mode="after")
    def check_if_path_exists(self):
        if not os.path.exists(self.target):
            msg = f"[ERROR] Path {self.target} not found."
            raise OSError(msg)
        return self


class HIPCutout(BaseModel):
    raTrim: PositiveFloat
    decTrim: PositiveFloat
    band: str


class HIPIntegrate(BaseModel):
    band: str
    radius: PositiveFloat


class HIPForegroundMask(BaseModel):
    maskFactor: PositiveFloat
    raTrim: PositiveFloat
    decTrim: PositiveFloat
    band: str


class HIPTest(BaseModel):
    pass


class HIPRms(BaseModel):
    band: str


Interface: Dict[str, BaseModel] = {
    "hip.degrade": HIPDegrade,
    "hip.skySubtract": HIPSkySubtract,
    "hip.regrid": HIPRegrid,
    "hip.cutout": HIPCutout,
    "hip.integrate": HIPIntegrate,
    "hip.foregroundMask": HIPForegroundMask,
    "hip.rms": HIPRms,
    "hip.test": HIPTest,
}
