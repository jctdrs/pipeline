import multiprocessing
import abc
import typing
import yaml
from astropy.io import fits

from hip import convolution
from hip import test
from hip import background
from hip import cutout
from hip import reproject
from hip import conversion

from util import file_manager

Interface: dict = {
    "hip.convolution": convolution.Convolution,
    "hip.test": test.Test,
    "hip.background": background.Background,
    "hip.cutout": cutout.Cutout,
    "hip.reproject": reproject.Reproject,
}


class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        self.file_mng = file_mng
        self.geom = file_mng.data["geometry"]
        self.load_instruments()

    @classmethod
    def create(cls, file_mng: file_manager.FileManager):
        parallel: bool = file_mng.config["parallel"]
        if parallel:
            return PipelineParallel(file_mng)
        else:
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
    ) -> typing.Union[fits.hdu.image.PrimaryHDU, typing.Optional[typing.Any]]:
        if "error" not in file_mng.data["bands"][idx]:
            return None

        err_path = f"{file_mng.data['bands'][idx]['error']}"
        hdul = fits.open(err_path)
        hdu = hdul[0]
        return hdu

    def load_instruments(self) -> None:
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
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        super().__init__(file_mng)
        self.result: typing.List[
            typing.Tuple[
                fits.hdu.image.PrimaryHDU, typing.Union[fits.hdu.image.PrimaryHDU, typing.Optional[typing.Any]]
            ]
        ] = []

    def execute(self) -> list:
        bands: dict = self.file_mng.data["bands"]
        for idx, _ in enumerate(bands):
            self.result.append((None, None))
            self._target(idx)
        return self.result

    def _target(self, idx: int) -> None:
        data_hdu = self.load_input(self.file_mng, idx)
        err_hdu = self.load_error(self.file_mng, idx)
        for task in self.file_mng.pipeline:
            data_hdu, err_hdu = Interface[task["step"]](
                data_hdu, err_hdu, self.geom, self.instruments, **task["parameters"]
            ).run()
        self.result[idx] = (data_hdu, err_hdu)
        return


class PipelineParallel(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        super().__init__(file_mng)
        self.processes: list = []
        self.result: typing.List[multiprocessing.Queue] = []

    def execute(self) -> list:
        # Spawn a process for every set of bands
        bands = self.file_mng.data["bands"]
        for idx, _ in enumerate(bands):
            self.result.append(multiprocessing.Queue())
            p = multiprocessing.Process(target=self._target, args=(idx,))
            self.processes.append(p)
            p.start()

        for process in self.processes:
            process.join()

        # self.result = [queue.get() for queue in self.result]
        return self.result

    def _target(self, idx: int) -> None:
        data_hdu = self.load_input(self.file_mng, idx)
        err_hdu = self.load_error(self.file_mng, idx)
        for task in self.file_mng.pipeline:
            data_hdu, err_hdu = Interface[task["step"]](
                data_hdu, err_hdu, self.geom, self.instruments, **task["parameters"]
            ).run()
        self.result[idx].put((data_hdu, err_hdu))
        return
