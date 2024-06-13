import abc
import typing
import yaml

from astropy.io import fits

import numpy as np

from hip import convolution
from hip import background
from hip import cutout
from hip import reproject
from hip import integrate
from hip import plot

from util import file_manager

Interface: dict = {
    "hip.convolution": convolution.Convolution,
    "hip.background": background.Background,
    "hip.cutout": cutout.Cutout,
    "hip.reproject": reproject.Reproject,
    "hip.integrate": integrate.Integrate,
    "hip.plot": plot.Plot,
}


class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager):
        self.file_mng = file_mng
        self.geom = file_mng.data["geometry"]
        self.result: typing.List[
            typing.Tuple[
                fits.hdu.image.PrimaryHDU,
                typing.Union[
                    fits.hdu.image.PrimaryHDU, typing.Optional[typing.Any]
                ],
            ]
        ] = []
        self.load_instruments()
        self.set_error_method()

    @classmethod
    def create(cls, file_mng: file_manager.FileManager) -> "PipelineSequential":
        return PipelineSequential(file_mng)

    @classmethod
    def load_input(
        cls, file_mng: file_manager.FileManager, idx: int
    ) -> fits.hdu.image.PrimaryHDU:
        inp_path = f"{file_mng.data['bands'][idx]['input']}"
        hdul = fits.open(inp_path)
        hdu = hdul[0]
        return hdu

    @classmethod
    def load_error(
        cls, file_mng: file_manager.FileManager, idx: int
    ) -> typing.Union[typing.Any, np.ndarray]:
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
            print(
                f"[ERROR]\tYAML parsing error at line {line}, column {column}."
            )
            exit()

    def set_error_method(self) -> None:
        if self.file_mng.config["error"] == "differr":
            self.use_jax = True
        else:
            self.use_jax = False

    @abc.abstractmethod
    def execute(self) -> list:
        pass


class PipelineSequential(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)

    def execute(self) -> list:
        bands: dict = self.file_mng.data["bands"]
        # Loop over bands
        for idx, _ in enumerate(bands):
            self._target(idx)
        return self.result

    def _target(self, idx: int) -> typing.Any:
        data_hdu = self.load_input(self.file_mng, idx)
        err_hdu = self.load_error(self.file_mng, idx)
        first_step_with_grad: bool = True

        # Loop over steps in pipeline
        for task in self.file_mng.pipeline:
            data_hdu, err_hdu, grad_arr = Interface[task["step"]](
                data_hdu,
                err_hdu,
                self.geom,
                self.instruments,
                self.use_jax,
                **task["parameters"],
            ).run()

            # Accumulate gradient
            if grad_arr is not None:
                if first_step_with_grad:
                    pipeline_grad = grad_arr
                    first_step_with_grad = False
                else:
                    pipeline_grad = np.tensordot(
                        grad_arr, pipeline_grad, axes=([2, 3], [0, 1])
                    )

        if not first_step_with_grad:
            import matplotlib.pyplot as plt

            plt.imshow(
                np.sqrt(
                    np.einsum("ijkl,kl->ij", pipeline_grad**2, err_hdu.data**2)
                ),
                origin="lower",
            )
            plt.xticks([])
            plt.yticks([])
            plt.grid()
            plt.show()

        self.result.append((data_hdu, None))
        return None
