import os
from typing import Optional

from utilities import read

from pydantic import BaseModel
from pydantic import NonNegativeFloat
from pydantic import model_validator

from astropy.io import fits


class Band(BaseModel):
    input: str
    output: str
    name: str
    error: Optional[str] = None
    calError: Optional[NonNegativeFloat] = None
    pixelSize: Optional[NonNegativeFloat] = None
    resolution: Optional[NonNegativeFloat] = None

    @model_validator(mode="after")
    def validate_paths(self):
        def check_if_path_exists(path):
            if not os.path.exists(path):
                msg = f"[ERROR] Path {path} not found."
                raise ValueError(msg)

        check_if_path_exists(self.input)
        check_if_path_exists(self.output)
        if self.error is not None:
            check_if_path_exists(self.error)
        return self

    @model_validator(mode="after")
    def fill_default_values_if_dustpedia_band(self):
        if self.name in DEFAULT_DUSTPEDIA_BANDS:
            band = DEFAULT_DUSTPEDIA_BANDS[self.name]
            if self.calError is None:
                self.calError = band.calError

            if self.resolution is None:
                self.resolution = band.resolution

        else:
            if self.calError is None:
                self.calError = 0
                msg = f"[WARNING] Calibration error not defined for {self.name}. Assumed null."
                print(msg)

            if self.resolution is None:
                msg = f"[ERROR] Resolution not defined for {self.name}."
                print(msg)
                raise ValueError(msg)

        if self.pixelSize is None:
            with fits.open(self.input) as hdu:
                pixel_size = read.pixel_size_arcsec(hdu[0].header)
                self.pixelSize = pixel_size

        return self


class DefaultBand(BaseModel):
    calError: float
    pixelSize: float
    resolution: float


DEFAULT_DUSTPEDIA_BANDS = {
    # GALEX
    "GALEX_FUV": DefaultBand(
        name="GALEX_FUV", resolution=4.3, pixelSize=3.2, calError=4.5
    ),
    "GALEX_NUV": DefaultBand(
        name="GALEX_NUV", resolution=5.3, pixelSize=3.2, calError=2.7
    ),
    #
    # SDSS
    "SDSS1": DefaultBand(name="SDSS1", resolution=1.3, pixelSize=0.45, calError=1.3),
    "SDSS2": DefaultBand(name="SDSS2", resolution=1.3, pixelSize=0.45, calError=0.8),
    "SDSS3": DefaultBand(name="SDSS3", resolution=1.3, pixelSize=0.45, calError=0.8),
    "SDSS4": DefaultBand(name="SDSS4", resolution=1.3, pixelSize=0.45, calError=0.7),
    "SDSS5": DefaultBand(name="SDSS5", resolution=1.3, pixelSize=0.45, calError=0.8),
    #
    # 2MASS
    "2MASS1": DefaultBand(name="2MASS1", resolution=2.0, pixelSize=1, calError=1.7),
    "2MASS2": DefaultBand(name="2MASS2", resolution=2.0, pixelSize=1, calError=1.9),
    "2MASS3": DefaultBand(name="2MASS3", resolution=2.0, pixelSize=1, calError=1.9),
    #
    # WISE
    "WISE1": DefaultBand(name="WISE1", resolution=6.1, pixelSize=1.375, calError=3),
    "WISE2": DefaultBand(name="WISE2", resolution=6.4, pixelSize=1.375, calError=3),
    "WISE3": DefaultBand(name="WISE3", resolution=6.5, pixelSize=1.375, calError=3),
    "WISE4": DefaultBand(name="WISE4", resolution=12, pixelSize=1.375, calError=3),
    #
    # Spitzer
    "IRAC1": DefaultBand(name="IRAC1", resolution=1.66, pixelSize=0.75, calError=3),
    "IRAC2": DefaultBand(name="IRAC2", resolution=1.72, pixelSize=0.75, calError=3),
    "IRAC3": DefaultBand(name="IRAC3", resolution=1.88, pixelSize=0.6, calError=3),
    "IRAC4": DefaultBand(name="IRAC4", resolution=1.98, pixelSize=0.6, calError=3),
    "MIPS1": DefaultBand(name="MIPS1", resolution=6, pixelSize=2.5, calError=5),
    "MIPS2": DefaultBand(name="MIPS2", resolution=18, pixelSize=4, calError=10),
    "MIPS3": DefaultBand(name="MIPS3", resolution=38, pixelSize=8, calError=12),
    #
    # Herschel
    "PACS1": DefaultBand(name="PACS1", resolution=9, pixelSize=2, calError=7),
    "PACS2": DefaultBand(name="PACS2", resolution=10, pixelSize=3, calError=7),
    "PACS3": DefaultBand(name="PACS3", resolution=13, pixelSize=4, calError=7),
    "SPIRE1": DefaultBand(name="SPIRE1", resolution=18, pixelSize=6, calError=5.5),
    "SPIRE2": DefaultBand(name="SPIRE2", resolution=25, pixelSize=8, calError=5.5),
    "SPIRE3": DefaultBand(name="SPIRE3", resolution=36, pixelSize=12, calError=5.5),
    #
    # Planck
    "HFI1": DefaultBand(name="HFI1", resolution=278, pixelSize=102.1, calError=6.4),
    "HFI2": DefaultBand(name="HFI2", resolution=290, pixelSize=102.1, calError=6.1),
    "HFI3": DefaultBand(name="HFI3", resolution=296, pixelSize=102.1, calError=0.78),
    "HFI4": DefaultBand(name="HFI4", resolution=301, pixelSize=102.1, calError=0.16),
    "HFI5": DefaultBand(name="HFI5", resolution=438, pixelSize=102.1, calError=0.07),
    "LFI1": DefaultBand(name="LFI1", resolution=581, pixelSize=205.7, calError=0.09),
    "LFI2": DefaultBand(name="LFI2", resolution=799, pixelSize=205.7, calError=0.20),
    "LFI3": DefaultBand(name="LFI3", resolution=1630, pixelSize=205.7, calError=0.26),
    "LFI4": DefaultBand(name="LFI4", resolution=1940, pixelSize=205.7, calError=0.35),
    #
    # IRAM30m
    "NIKA2_1": DefaultBand(name="NIKA2_1", resolution=12, pixelSize=3, calError=5),
    "NIKS2_2": DefaultBand(name="NIKA2_2", resolution=18, pixelSize=4, calError=5),
}
