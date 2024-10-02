import typing
import yaml

from astropy.io import fits

import numpy as np

from hip import convolution
from hip import background
from hip import reproject
from hip import cutout
from hip import foreground

from util import file_manager
from util import plot
from util import integrate

Interface: dict = {
    "hip.convolution": convolution.Convolution,
    "hip.background": background.Background,
    "hip.reproject": reproject.Reproject,
    "hip.cutout": cutout.Cutout,
    "util.integrate": integrate.Integrate,
    "util.plot": plot.Plot,
    "hip.foreground": foreground.Foreground,
}


class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager):
        self.file_mng = file_mng
        self.geom = file_mng.data["geometry"]
        self.result: typing.List[
            typing.Tuple[fits.hdu.image.PrimaryHDU, fits.hdu.image.PrimaryHDU]
        ] = []
        self.load_instruments()

    @classmethod
    def create(
        cls, file_mng: file_manager.FileManager
    ) -> typing.Union[
        "DifferentialPipeline", "MonteCarloPipeline", "SinglePassPipeline"
    ]:
        if file_mng.config["error"] == "differr":
            return DifferentialPipeline(file_mng)

        elif file_mng.config["error"] == "MC":
            niter = file_mng.config["iterations"]
            if niter == 1:
                return SinglePassPipeline(file_mng)
            elif niter > 1:
                return MonteCarloPipeline(file_mng)

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

    def save_output(self) -> typing.Any:
        bands: dict = self.file_mng.data["bands"]

        # Loop over bands
        for idx, band in enumerate(bands):
            fits.writeto(
                f"{band['name']}.fits",
                self.result[idx][0].data,
                self.result[idx][0].header,
                overwrite=True,
            )
            if self.result[idx][1] is not None:
                fits.writeto(
                    f"{band['name']}_Error.fits",
                    self.result[idx][1].data,
                    self.result[idx][1].header,
                    overwrite=True,
                )

    def execute(self) -> list:
        bands: dict = self.file_mng.data["bands"]
        # Loop over bands
        for idx, _ in enumerate(bands):
            self._target(idx)
        return self.result

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


class DifferentialPipeline(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)

    def _target(self, idx: int) -> typing.Any:
        data_hdu = self.load_input(self.file_mng, idx)
        err_hdu = self.load_input(self.file_mng, idx)

        first_step_with_grad: bool = True
        pipeline_grad = None

        body = self.file_mng.data["body"]
        name = self.file_mng.data["bands"][idx]["name"]
        # Loop over steps in pipeline
        for task in self.file_mng.pipeline:
            data_hdu, err_hdu, grad_arr = Interface[task["step"]](
                data_hdu,
                name,
                body,
                self.geom,
                self.instruments,
                True,
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

        # TODO: Make sure err_hdu.data has the correct shape
        if pipeline_grad is not None:
            final = np.sqrt(np.einsum("ijkl,kl->ij", pipeline_grad**2, err_hdu.data**2))  # noqa

        self.result.append((data_hdu, err_hdu))
        self.save_output()
        return None


class SinglePassPipeline(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)

    def _target(self, idx: int) -> typing.Any:
        body = self.file_mng.data["body"]
        name = self.file_mng.data["bands"][idx]["name"]

        data_hdu = self.load_input(self.file_mng, idx)
        err_hdu = None

        for task in self.file_mng.tasks:
            print(task)
            data_hdu, _ = Interface[task["step"]](
                data_hdu,
                name,
                body,
                self.geom,
                self.instruments,
                False,
                **task["parameters"],
            ).run()

        self.result.append((data_hdu, err_hdu))
        self.save_output()
        return None


class MonteCarloPipeline(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)
        self.niter = file_mng.config["iterations"]

    def _target(self, idx: int) -> typing.Any:
        # Loop over steps in pipeline
        count = 0
        mean = 0
        M2 = 0

        body = self.file_mng.data["body"]
        name = self.file_mng.data["bands"][idx]["name"]

        if "error" in self.file_mng.data["bands"][idx]:
            original_data_hdu = self.load_input(self.file_mng, idx)
            original_err_hdu = self.load_error(self.file_mng, idx)

            if hasattr(self.file_mng, "before"):
                for task in self.file_mng.before:
                    original_data_hdu, _ = Interface[task["step"]](
                        original_data_hdu,
                        name,
                        body,
                        self.geom,
                        self.instruments,
                        False,
                        **task["parameters"],
                    ).run()

            for niter in range(self.niter):
                print(niter)
                count += 1

                data_hdu = fits.PrimaryHDU(
                    header=original_data_hdu.header,
                    data=original_data_hdu.data
                    + np.random.normal(
                        0, original_err_hdu.data, original_err_hdu.data.shape
                    ),
                )

                for task in self.file_mng.pipeline:
                    data_hdu, _ = Interface[task["step"]](
                        data_hdu,
                        name,
                        body,
                        self.geom,
                        self.instruments,
                        False,
                        **task["parameters"],
                    ).run()

                # Running sum
                if niter == 0:
                    data_result = data_hdu.data
                else:
                    data_result += data_hdu.data

                # Running variance
                delta = data_hdu.data - mean
                mean += delta / count
                delta2 = data_hdu.data - mean
                M2 += delta * delta2

            # Mean
            data_hdu.data = data_result / count

            # Standard deviation
            err_hdu = fits.PrimaryHDU(
                header=data_hdu.header, data=np.sqrt(M2 / (count - 1))
            )

            if hasattr(self.file_mng, "after"):
                for task in self.file_mng.after:
                    data_hdu, _ = Interface[task["step"]](
                        data_hdu,
                        name,
                        body,
                        self.geom,
                        self.instruments,
                        False,
                        **task["parameters"],
                    ).run()

        self.result.append((data_hdu, err_hdu))
        self.save_output()
        return None
