import abc
import typing
import yaml
from astropy.io import fits

from hip import convolution
from hip import background
from hip import cutout
from hip import reproject
from hip import integrate

from util import file_manager

Interface: dict = {
    "hip.convolution": convolution.Convolution,
    "hip.background": background.Background,
    "hip.cutout": cutout.Cutout,
    "hip.reproject": reproject.Reproject,
}


class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager):
        self.file_mng = file_mng
        self.geom = file_mng.data["geometry"]
        self.result: typing.List[
            typing.Tuple[
                fits.hdu.image.PrimaryHDU, typing.Union[fits.hdu.image.PrimaryHDU, typing.Optional[typing.Any]]
            ]
        ] = []
        self.load_instruments()

    @classmethod
    def create(cls, file_mng: file_manager.FileManager) -> "PipelineSequential":
        return PipelineSequential(file_mng)

    @classmethod
    def load_input(cls, file_mng: file_manager.FileManager, idx: int) -> fits.hdu.image.PrimaryHDU:
        inp_path = f"{file_mng.data['bands'][idx]['input']}"
        hdul = fits.open(inp_path)
        hdu = hdul[0]
        return hdu

    @classmethod
    def load_error(
        cls, file_mng: file_manager.FileManager, idx: int
    ) -> typing.Union[typing.Any, fits.hdu.image.PrimaryHDU]:
        if "error" not in file_mng.data["bands"][idx]:
            return None

        err_path = f"{file_mng.data['bands'][idx]['error']}"
        hdul = fits.open(err_path)
        hdu = hdul[0]
        return hdu

    def load_instruments(self) -> typing.Any:
        try:
            f = open("config/instruments.yml", "r")
        except OSError:
            print("[ERROR]\tFile 'config/instruments.yml' not found.")
            exit()

        # Check if specification is valid YAML
        # In case of failure, capture line/col for debug
        try:
            self.instruments = yaml.load(f, yaml.SafeLoader)
            f.close()
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            line = e.problem_mark.line + 1  # type: ignore
            column = e.problem_mark.column + 1  # type: ignore
            print(f"[ERROR]\tYAML parsing error at line {line}, column {column}.")
            exit()

    @abc.abstractmethod
    def execute(self) -> list:
        pass


class PipelineSequential(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)

    def execute(self) -> list:
        bands: dict = self.file_mng.data["bands"]
        for idx, _ in enumerate(bands):
            self.result.append((None, None))
            self._target(idx)
        return self.result

    def _target(self, idx: int) -> typing.Any:
        data_hdu = self.load_input(self.file_mng, idx)
        err_hdu = self.load_error(self.file_mng, idx)
        for task in self.file_mng.pipeline:
            data_hdu, err_hdu = Interface[task["step"]](
                data_hdu, err_hdu, self.geom, self.instruments, **task["parameters"]
            ).run()

        # Add last step: Integration
        fluxes = integrate.Integrate(data_hdu, err_hdu, self.geom, self.instruments).run()
        self.result[idx] = (data_hdu, err_hdu)
        print(f"[DEBUG]\tIntegrated fluxes {fluxes[2]}")
        return None
