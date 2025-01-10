import os
from typing import Optional

from pydantic import BaseModel
from pydantic import NonNegativeFloat
from pydantic import model_validator

DEFAULT_CALIBRATION_ERROR = {
    "IRAC1": 10.2,
    "IRAC2": 10.2,
    "IRAC3": 10.2,
    "IRAC4": 10.2,
    "WISE1_ATLAS": 3.2,
    "WISE2_ATLAS": 3.5,
    "WISE3_ATLAS": 5.0,
    "WISE4_ATLAS": 7.0,
    "IPS1": 4.0,
    "IPS2": 5.0,
    "IPS3": 11.6,
    "PACS1": 5.4,
    "PACS2": 5.4,
    "PACS3": 5.4,
    "SPIRE1": 5.9,
    "SPIRE2": 5.9,
    "SPIRE3": 5.9,
    "NIKA2_1": 5.0,
    "NIKA2_2": 5.0,
    "HFI1": 4.3,
    "HFI2": 4.2,
    "HFI3": 0.9,
    "HFI4": 0.9,
    "HFI5": 0.9,
    "2MASS1": 1.7,
    "2MASS2": 1.9,
    "2MASS3": 1.9,
    "SDSS1": 1.3,
    "SDSS2": 0.8,
    "SDSS3": 0.8,
    "SDSS4": 0.7,
    "SDSS5": 0.8,
    "GALEX_FUV": 4.5,
    "GALEX_NUV": 2.7,
}


class Band(BaseModel):
    input: str
    output: str
    name: str
    error: Optional[str] = None
    calError: Optional[NonNegativeFloat] = None

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
        if self.calError is None:
            try:
                cal_error: float = DEFAULT_CALIBRATION_ERROR[self.name]
                msg = f"[WARNING] Calibration error not defined, assuming {cal_error} for {self.name}."
            except KeyError:
                cal_error: float = 0.0
                msg = "[WARNING] Calibration error not defined, asumming null."

            self.calError = cal_error
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
