import os
from typing_extensions import Optional

from pydantic import BaseModel
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator


# A factory method design to return the correct validator for each step of the
# pipeline depending on what the user needs. Each validator is unique to the
# nature of the step.
def factory_method(pipeline_step: str, **parameters):
    if pipeline_step not in Interface:
        msg = f"[ERROR] Pipeline step {pipeline_step} not valid."
        raise ValueError(msg)
    pipeline_step_class = Interface.get(pipeline_step)
    return pipeline_step_class(**parameters)


class HIPDegrade(BaseModel):
    target: Optional[str] = None
    kernel: Optional[str] = None
    type: Optional[str] = None
    name: str
    band: str

    @model_validator(mode="after")
    def check_if_kernel_path_exists(self):
        if self.kernel is not None:
            if not os.path.exists(self.kernel):
                msg = f"[ERROR] Path {self.kernel} not found."
                raise OSError(msg)
        return self

    @model_validator(mode="after")
    def check_if_target_path_exists(self):
        if self.target is not None:
            if not os.path.exists(self.target):
                msg = f"[ERROR] Path {self.target} not found."
                raise OSError(msg)
        return self

    @model_validator(mode="after")
    def check_if_type_is_defined_when_target_is_defined(self):
        if self.target is not None and self.type is None:
            self.type = "Gaussian"
            msg = "[WARNING] 'Type' not defined, defaulting to 'Gaussian'."
            print(msg)
        return self

    @model_validator(mode="after")
    def check_type(self):
        valid_type = {"Gaussian", "Aniano"}
        if self.type is not None and self.type not in valid_type:
            msg = f"[ERROR] 'Type' {self.type} not valid."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def ignore_type_if_kernel_defined(self):
        if self.kernel is not None and self.type is not None:
            msg = "[WARNING] Ignoring 'type' when 'kernel' is defined."
            print(msg)
        return self

    @model_validator(mode="after")
    def check_if_kernel_and_target_are_defined(self):
        if self.kernel is not None and self.target is not None:
            msg = "[ERROR] Both 'target' and 'kernel' are defined."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_herbie_band_names(self):
        if (
            self.name != "IRAC1"
            and self.name != "IRAC2"
            and self.name != "IRAC3"
            and self.name != "IRAC4"
            and self.name != "WISE1_ATLAS"
            and self.name != "WISE2_ATLAS"
            and self.name != "WISE3_ATLAS"
            and self.name != "WISE3_ATLAS"
            and self.name != "WISE4_ATLAS"
            and self.name != "MIPS1"
            and self.name != "MIPS2"
            and self.name != "MIPS3"
            and self.name != "PACS1"
            and self.name != "PACS2"
            and self.name != "PACS3"
            and self.name != "SPIRE1"
            and self.name != "SPIRE2"
            and self.name != "SPIRE3"
            and self.name != "NIKA2_1"
            and self.name != "NIKA2_2"
            and self.name != "HFI1"
            and self.name != "HFI2"
            and self.name != "HFI3"
            and self.name != "HFI4"
            and self.name != "HFI5"
            and self.name != "SDSS1"
            and self.name != "SDSS2"
            and self.name != "SDSS3"
            and self.name != "SDSS4"
            and self.name != "SDSS5"
            and self.name != "2MASS1"
            and self.name != "2MASS2"
            and self.name != "2MASS3"
            and self.name != "GALEX_FUV"
            and self.name != "GALEX_NUV"
        ):
            msg = f"[Error] Band '{self.name}' not valid HerBie naming."
            raise ValueError(msg)
        return self


class HIPSkySubtract(BaseModel):
    cellSize: Optional[PositiveInt] = 1
    band: str


class HIPRegrid(BaseModel):
    target: str
    name: str
    band: str

    @model_validator(mode="after")
    def check_if_path_exists(self):
        if not os.path.exists(self.target):
            msg = f"[ERROR] Path {self.target} not found."
            raise OSError(msg)
        return self

    @model_validator(mode="after")
    def validate_herbie_band_names(self):
        if (
            self.name != "IRAC1"
            and self.name != "IRAC2"
            and self.name != "IRAC3"
            and self.name != "IRAC4"
            and self.name != "WISE1_ATLAS"
            and self.name != "WISE2_ATLAS"
            and self.name != "WISE3_ATLAS"
            and self.name != "WISE3_ATLAS"
            and self.name != "WISE4_ATLAS"
            and self.name != "MIPS1"
            and self.name != "MIPS2"
            and self.name != "MIPS3"
            and self.name != "PACS1"
            and self.name != "PACS2"
            and self.name != "PACS3"
            and self.name != "SPIRE1"
            and self.name != "SPIRE2"
            and self.name != "SPIRE3"
            and self.name != "NIKA2_1"
            and self.name != "NIKA2_2"
            and self.name != "HFI1"
            and self.name != "HFI2"
            and self.name != "HFI3"
            and self.name != "HFI4"
            and self.name != "HFI5"
            and self.name != "SDSS1"
            and self.name != "SDSS2"
            and self.name != "SDSS3"
            and self.name != "SDSS4"
            and self.name != "SDSS5"
            and self.name != "2MASS1"
            and self.name != "2MASS2"
            and self.name != "2MASS3"
            and self.name != "GALEX_FUV"
            and self.name != "GALEX_NUV"
        ):
            msg = f"[Error] Band '{self.name}' not valid HerBie naming."
            raise ValueError(msg)
        return self


class HIPCutout(BaseModel):
    raTrim: PositiveFloat
    decTrim: PositiveFloat
    band: str


class HIPIntegrate(BaseModel):
    band: str
    radius: PositiveFloat


class HIPForegroundMask(BaseModel):
    factor: PositiveFloat
    raTrim: PositiveFloat
    decTrim: PositiveFloat
    band: str


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
