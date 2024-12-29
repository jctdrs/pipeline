import os
from typing_extensions import Optional

from pydantic import BaseModel
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator


def factory_method(pipeline_step: str, **parameters):
    if pipeline_step not in Interface:
        msg = f"[ERROR] Pipeline step {pipeline_step} not defined."
        raise ValueError(msg)
    pipeline_step_class = Interface.get(pipeline_step)
    return pipeline_step_class(**parameters)


class HIPDegrade(BaseModel):
    kernel: str

    @model_validator(mode="after")
    def check_if_path_exists(self):
        if not os.path.exists(self.kernel):
            msg = f"[ERROR] Path {self.kernel} not found."
            raise OSError(msg)
        return self


class HIPSkySubtract(BaseModel):
    cellSize: Optional[PositiveInt] = 1


class HIPRegrid(BaseModel):
    target: str

    @model_validator(mode="after")
    def check_if_path_exists(self):
        if not os.path.exists(self.target):
            msg = f"[ERROR] Path {self.target} not found."
            raise OSError(msg)
        return self


class HIPCutout(BaseModel):
    raTrim: PositiveFloat
    decTrim: PositiveFloat


class HIPIntegrate(BaseModel):
    radius: PositiveFloat


class HIPForegroundMask(BaseModel):
    factor: PositiveFloat
    raTrim: PositiveFloat
    decTrim: PositiveFloat


class HIPTest(BaseModel):
    pass


Interface = {
    "hip.degrade": HIPDegrade,
    "hip.skySubtract": HIPSkySubtract,
    "hip.regrid": HIPRegrid,
    "hip.cutout": HIPCutout,
    "hip.integrate": HIPIntegrate,
    "hip.foregroundMask": HIPForegroundMask,
    "hip.test": HIPTest,
}
