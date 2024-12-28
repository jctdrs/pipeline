import os

from pydantic import BaseModel, PositiveFloat, PositiveInt, model_validator  # noqa
from typing_extensions import Optional


def factory_method(pipeline_step: str, **parameters):
    if pipeline_step not in Interface:
        msg = f"[ERROR] Pipeline step {pipeline_step} not defined."
        raise ValueError(msg)
    pipeline_step_class = Interface.get(pipeline_step)
    return pipeline_step_class(**parameters)


class HIPConvolutionValidation(BaseModel):
    kernel: str

    @model_validator(mode="after")
    def check_if_path_exists(self):
        if not os.path.exists(self.kernel):
            msg = f"[ERROR] Path {self.kernel} not found."
            raise OSError(msg)
        return self


class HIPSkySubtractionValidation(BaseModel):
    cellSize: Optional[PositiveInt] = 1


class HIPReprojectValidation(BaseModel):
    target: str

    @model_validator(mode="after")
    def check_if_path_exists(self):
        if not os.path.exists(self.target):
            msg = f"[ERROR] Path {self.target} not found."
            raise OSError(msg)
        return self


class HIPCutoutValidation(BaseModel):
    raTrim: PositiveFloat
    decTrim: PositiveFloat


class HIPPhotometryValidation(BaseModel):
    radius: PositiveFloat


class HIPForegroundMaskingValidation(BaseModel):
    factor: PositiveFloat
    raTrim: PositiveFloat
    decTrim: PositiveFloat


class HIPTestValidation(BaseModel):
    pass


Interface = {
    "hip.convolution": HIPConvolutionValidation,
    "hip.skySubtraction": HIPSkySubtractionValidation,
    "hip.reproject": HIPReprojectValidation,
    "hip.cutout": HIPCutoutValidation,
    "hip.photometry": HIPPhotometryValidation,
    "hip.foregounrdMasking": HIPForegroundMaskingValidation,
    "hip.test": HIPTestValidation,
}
