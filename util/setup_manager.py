import os
import yaml
import typing

from util import file_manager


class SetupManager:
    def __init__(self, file_mng: file_manager.FileManager):
        self.file_mng = file_mng

    def set(self):
        self.validate_files()
        self.validate_input_bands()
        self.validate_convolution()

    def validate_files(self) -> typing.Any:
        status: bool = True
        files_not_found: str = ""
        bands: dict = self.file_mng.data["bands"]

        # Check that files in specification exist for all
        # bodies, all bands, and all input and error files
        for band in bands:
            for key in band.keys():
                if key == "input" or key == "error":
                    filename: str = f"{band[key]}"
                    if not os.path.exists(filename):
                        files_not_found += f" '{filename}',"
                        status = False

        if not status:
            print(f"[ERROR]\tFile(s) not found{files_not_found}.")
            exit()
        return None

    def validate_input_bands(self) -> typing.Any:
        bands: dict = self.file_mng.data["bands"]
        for band in bands:
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
        target_bands: list = []
        kernels: list = []
        for step in pipeline:
            if step["step"] == "hip.convolution":
                target_band = step["parameters"]["name"]
                kernel = step["parameters"]["kernel"]
                self.validate_band(target_band)
                target_bands.append(target_band)
                kernels.append(kernel)

        self.validate_resolution(target_bands)
        self.check_for_kernels(kernels)

    def validate_resolution(self, target_bands: list) -> typing.Any:
        try:
            f = open("config/instruments.yml", "r")
        except OSError:
            print("[ERROR]\tFile 'config/instruments.yml' not found.")
            exit()

        # Check if specification is valid YAML
        # In case of failure, capture line/col for debug
        try:
            instruments = yaml.load(f, yaml.SafeLoader)
            f.close()
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            line = e.problem_mark.line + 1  # type: ignore
            column = e.problem_mark.column + 1  # type: ignore
            print(
                f"[ERROR]\tYAML parsing error at line {line}, column {column}."
            )
            exit()

        bands = self.file_mng.data["bands"]
        status: bool = True
        bad_band: str = ""
        bad_target: str = ""
        for target_band in target_bands:
            target_resolution = instruments[target_band]["RESOLUTION"]["VALUE"]
            for band in bands:
                band_name = band["name"]
                band_resolution = instruments[band_name]["RESOLUTION"]["VALUE"]
                if target_resolution < band_resolution:
                    status = False
                    bad_band += f" '{band_name}'"
                    bad_target += f" '{target_band}'"

        if not status:
            print(
                f"[ERROR]\tCannot implement resolution degradation from{bad_band} to{bad_target}."
            )
            exit()

    def check_for_kernels(self, kernels: list) -> typing.Any:
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
