import os
from typing import Optional
from typing import Any

from pydantic import BaseModel
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator


# A factory method design to return the correct validator for each step of the
# pipeline depending on what the user needs. Each validator is unique to the
# nature of the step.
def factory_method(pipeline_step: str, **parameters) -> Any:
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
    sizeFactor: Optional[PositiveFloat] = 1.0
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
    sizeFactor: Optional[PositiveFloat] = 1.0


class HIPForegroundMask(BaseModel):
    maskFactor: Optional[PositiveFloat] = 1.0
    raTrim: PositiveFloat
    decTrim: PositiveFloat
    band: str


class HIPTest(BaseModel):
    pass

class CoreSubtract(BaseModel):
    band: str
    target: Optional[str] = None
    resultOf: Optional[str] = None
    factor: Optional[float] = None

    @model_validator(mode="after")
    def check_for_only_one_option_defined(self):
        vars_dict = {"target": self.target, "resultOf": self.resultOf, "factor": self.factor}
        defined = [name for name, value in vars_dict.items() if value is not None]
        if len(defined) > 1:
            msg = f"More than one option is defined: {defined}"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def check_if_target_path_exists(self):
        if self.target is not None:
            if not os.path.exists(self.target):
                msg = f"[ERROR] Path {self.target} not found."
                raise OSError(msg)
        return self

    @model_validator(mode="after")
    def check_if_resultOf_path_exists(self):
        if self.resultOf is not None:
            if not os.path.exists(self.resultOf):
                msg = f"[ERROR] Path {self.resultOf} not found."
                raise OSError(msg)
        return self

class CoreMultiply(BaseModel):
    band: str
    target: Optional[str] = None
    resultOf: Optional[str] = None
    factor: Optional[float] = None

    @model_validator(mode="after")
    def check_for_only_one_option_defined(self):
        vars_dict = {"target": self.target, "resultOf": self.resultOf, "factor": self.factor}
        defined = [name for name, value in vars_dict.items() if value is not None]
        if len(defined) > 1:
            msg = f"More than one option is defined: {defined}"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def check_if_target_path_exists(self):
        if self.target is not None:
            if not os.path.exists(self.target):
                msg = f"[ERROR] Path {self.target} not found."
                raise OSError(msg)
        return self

    @model_validator(mode="after")
    def check_if_resultOf_path_exists(self):
        if self.resultOf is not None:
            if not os.path.exists(self.resultOf):
                msg = f"[ERROR] Path {self.resultOf} not found."
                raise OSError(msg)
        return self

Interface = {
    "hip.degrade": HIPDegrade,
    "hip.skySubtract": HIPSkySubtract,
    "hip.regrid": HIPRegrid,
    "hip.cutout": HIPCutout,
    "hip.integrate": HIPIntegrate,
    "hip.foregroundMask": HIPForegroundMask,
    "hip.test": HIPTest,
    "core.subtract": CoreSubtract,
    "core.multiply": CoreMultiply,
}
