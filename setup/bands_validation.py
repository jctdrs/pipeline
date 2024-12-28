import os

from pydantic import BaseModel, PositiveFloat, model_validator
from typing import Optional


class Bands(BaseModel):
    input: str
    output: str
    name: str
    error: Optional[str] = None
    calError: Optional[PositiveFloat] = 0.0

    @model_validator(mode="after")
    def validate_paths(self):
        def check_if_path_exists(path):
            if not os.path.exists(path):
                msg = f"[ERROR] Path {path} not found."
                raise OSError(msg)

        check_if_path_exists(self.input)
        check_if_path_exists(self.output)
        if self.error is not None:
            check_if_path_exists(self.error)
        return self

    @model_validator(mode="after")
    def warning_if_calibration_error_not_defined(self):
        if self.calError == 0:
            msg = "[WARNING] Calibration error not defined, assuming null."
            print(msg)

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
