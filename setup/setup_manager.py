import os
import typing
import csv

from setup import file_manager

PHOTOMETRY_CONFIG = "config/DustPedia_Aperture_Photometry_2.2.csv"
DISTANCES_CONFIG = "config/DustPedia_HyperLEDA_Herschel.csv"


class SetupManager:
    def __init__(self, file_mng: file_manager.FileManager):
        self.file_mng = file_mng

    def set(self):
        self.validate_files()
        self.validate_body()
        self.validate_input_band()
        self.validate_convolution()

    def validate_files(self) -> typing.Any:
        status: bool = True
        files_not_found: str = ""
        band: dict = self.file_mng.data["band"]

        # Check that files in specification exist for all
        # bodies, all band, and all input and error files
        for key in band:
            if key == "input" or key == "error":
                filename: str = f"{band[key]}"
                if not os.path.exists(filename):
                    files_not_found += f" '{filename}',"
                    status = False

        if not status:
            print(f"[ERROR]\tFile(s) not found{files_not_found}.")
            exit()
        return None

    def validate_body(self) -> typing.Any:
        body = self.file_mng.data["body"]

        required_photometry = {
            "ra": "ra",
            "dec": "dec",
            "positionAngle": "pos_angle",
            # "distance"
            # "redshift",
            "axialRatio": "axial_ratio",
            "semiMajorAxis": "semimaj_arcsec",
            # "inclination",
            # "radius",
        }

        with open(PHOTOMETRY_CONFIG) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] == body:
                    for key, value in required_photometry.items():
                        if key not in self.file_mng.data["geometry"].keys():
                            self.file_mng.data["geometry"][key] = float(row[value])
                    break

        required_distances = {"inclination": "incl", "radius": "d25"}

        with open(DISTANCES_CONFIG) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["objname"] == body:
                    for key, value in required_distances.items():
                        if key not in self.file_mng.data["geometry"].keys():
                            self.file_mng.data["geometry"][key] = float(row[value])
                    break

        return None

    def validate_input_band(self) -> typing.Any:
        band: dict = self.file_mng.data["band"]
        self.validate_band(band["name"])
        return None

    def validate_band(self, band) -> typing.Any:
        if (
            band != "IRAC1"
            and band != "IRAC2"
            and band != "IRAC3"
            and band != "IRAC4"
            and band != "WISE1_ATLAS"
            and band != "WISE2_ATLAS"
            and band != "WISE3_ATLAS"
            and band != "WISE3_ATLAS"
            and band != "WISE4_ATLAS"
            and band != "MIPS1"
            and band != "MIPS2"
            and band != "MIPS3"
            and band != "PACS1"
            and band != "PACS2"
            and band != "PACS3"
            and band != "SPIRE1"
            and band != "SPIRE2"
            and band != "SPIRE3"
            and band != "NIKA2_1"
            and band != "NIKA2_2"
            and band != "HFI1"
            and band != "HFI2"
            and band != "HFI3"
            and band != "HFI4"
            and band != "HFI5"
            and band != "SDSS1"
            and band != "SDSS2"
            and band != "SDSS3"
            and band != "SDSS4"
            and band != "SDSS5"
            and band != "2MASS1"
            and band != "2MASS2"
            and band != "2MASS3"
            and band != "GALEX_FUV"
            and band != "GALEX_NUV"
        ):
            print(f"[Error]\tBand '{band}' not valid")
            exit()
        return None

    def validate_convolution(self) -> typing.Any:
        pipeline = self.file_mng.pipeline
        kernels: list = []
        for step in pipeline:
            if step["step"] == "hip.convolution":
                kernel = step["parameters"]["kernel"]
                kernels.append(kernel)

        status: bool = True
        kernels_not_found: str = ""
        for kernel in kernels:
            if not os.path.exists(kernel):
                kernels_not_found += f" '{kernel}'"
                status = False

        if not status:
            print(f"[ERROR]\tKernel(s) not found{kernels_not_found}.")
            exit()

        return None
