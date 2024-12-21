import typing
import yaml
import copy

from astropy.io import fits
from astropy.stats import mad_std

import jax
import jax.numpy as jnp

from hip import convolution
from hip import background
from hip import reproject
from hip import foreground

from setup import file_manager

from util import plot
from util import integrate
from util import cutout
from util import test

INSTRUMENTS_CONFIG = "data/config/instruments.yml"

Interface: dict = {
    "hip.convolution": convolution.Convolution,
    "hip.background": background.Background,
    "hip.reproject": reproject.Reproject,
    "util.cutout": cutout.Cutout,
    "util.integrate": integrate.Integrate,
    "util.plot": plot.Plot,
    "hip.foreground": foreground.Foreground,
    "util.test": test.Test,
}


class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager):
        self.file_mng = file_mng
        self.geom = file_mng.data["geometry"]
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
            niter = file_mng.config["niter"]
            if niter == 1:
                return SinglePassPipeline(file_mng)
            elif niter > 1:
                SinglePassPipeline(file_mng).execute()
                return MonteCarloPipeline(file_mng)

    def load_input(self, file_mng: file_manager.FileManager) -> None:
        inp_path = f"{file_mng.data['band']['input']}"
        hdul = fits.open(inp_path)
        self.data_hdu = hdul[0]
        return

    def load_error(self, file_mng: file_manager.FileManager) -> None:
        if "error" not in file_mng.data["band"]:
            return

        err_path = f"{file_mng.data['band']['error']}"
        hdul = fits.open(err_path)
        self.err_hdu = hdul[0]
        return

    def save_data(self) -> typing.Any:
        band: dict = self.file_mng.data["band"]

        fits.writeto(
            f"{band['name']}.fits",
            self.data_hdu.data,
            self.data_hdu.header,
            overwrite=True,
        )

    def save_error(self) -> typing.Any:
        band: dict = self.file_mng.data["band"]

        if self.err_hdu is not None:
            fits.writeto(
                f"{band['name']}_Error.fits",
                self.err_hdu.data,
                self.err_hdu.header,
                overwrite=True,
            )

    def execute(self):
        pass

    def load_instruments(self) -> typing.Any:
        try:
            f = open(INSTRUMENTS_CONFIG, "r")
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


# TODO: If error not define in YAML then abort
class DifferentialPipeline(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)

    def execute(self) -> typing.Any:
        self.load_input(self.file_mng)

        if "error" in self.file_mng.data["band"]:
            self.load_error(self.file_mng)
        else:
            std_data = mad_std(self.data_hdu.data, ignore_nan=True)
            self.err_hdu = fits.PrimaryHDU(
                header=fits.Header(), data=jnp.full_like(self.data_hdu.data, std_data)
            )[0]

        first_step_with_grad: bool = True
        pipeline_grad = None

        body = self.file_mng.data["body"]
        name = self.file_mng.data["band"]["name"]

        # Loop over steps in pipeline
        for task in self.file_mng.tasks:
            self.data_hdu, self.err_hdu, grad_arr = Interface[task["step"]](
                self.data_hdu,
                self.err_hdu,
                name,
                body,
                self.geom,
                self.instruments,
                diagnosis=task["diagnosis"],
                MC_diagnosis=False,
                differentiate=True,
                **task["parameters"],
            ).run()

            # Accumulate gradient
            if grad_arr is not None:
                if first_step_with_grad:
                    pipeline_grad = grad_arr
                    first_step_with_grad = False
                else:
                    pipeline_grad = jnp.tensordot(
                        grad_arr, pipeline_grad, axes=([2, 3], [0, 1])
                    )

        if pipeline_grad is not None:
            self.err_hdu.data = jnp.sqrt(
                jnp.einsum("ijkl,kl->ij", pipeline_grad**2, self.err_hdu.data**2)
            )

        self.save_data()
        self.save_error()
        return None


class SinglePassPipeline(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)

    def execute(self) -> typing.Any:
        body = self.file_mng.data["body"]
        name = self.file_mng.data["band"]["name"]

        self.load_input(self.file_mng)
        self.err_hdu = None
        for task in self.file_mng.single:
            self.data_hdu, self.err_hdu, _ = Interface[task["step"]](
                self.data_hdu,
                self.err_hdu,
                name,
                body,
                self.geom,
                self.instruments,
                diagnosis=task["diagnosis"],
                MC_diagnosis=False,
                differentiate=False,
                **task["parameters"],
            ).run()

        self.save_data()
        return None


class MonteCarloPipeline(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager):
        super().__init__(file_mng)
        self.niter: int = file_mng.config["niter"]
        self.repeat: list = file_mng.repeat

    def execute(self) -> typing.Any:
        count: float = 0
        mean: float = 0
        M2: float = 0

        MC_diagnosis: bool = False

        key = jax.random.key(638)

        self.body = self.file_mng.data["body"]
        self.name = self.file_mng.data["band"]["name"]

        self.load_input(self.file_mng)

        if "error" in self.file_mng.data["band"]:
            self.load_error(self.file_mng)
        else:
            std_data = mad_std(self.data_hdu.data, ignore_nan=True)
            self.err_hdu = fits.PrimaryHDU(
                header=fits.Header(), data=jnp.full_like(self.data_hdu.data, std_data)
            )[0]

        for idx, task in enumerate(self.file_mng.tasks):
            # TODO: This only works if the task is at the very end. Should do for entire MC tasks
            if idx == len(self.file_mng.tasks) - 1:
                MC_diagnosis = True

            if self.repeat[idx] == 1 or self.repeat[idx] == 2:
                key, subkey = jax.random.split(key)

                if "original_data_hdu" not in locals():
                    original_data_hdu = self.data_hdu
                    original_err_hdu = self.err_hdu

                self.data_hdu = fits.PrimaryHDU(
                    header=original_data_hdu.header,
                    data=jnp.array(original_data_hdu.data)
                    + jnp.array(original_err_hdu.data)
                    * jax.random.normal(subkey, original_err_hdu.data.shape),
                )

                self.err_hdu = copy.copy(original_err_hdu)
                count += 1

            self.data_hdu, self.err_hdu, _ = Interface[task["step"]](
                self.data_hdu,
                self.err_hdu,
                self.name,
                self.body,
                self.geom,
                self.instruments,
                diagnosis=False,
                MC_diagnosis=MC_diagnosis,
                differentiate=False,
                **task["parameters"],
            ).run()

            # Running sum
            if self.repeat[idx] == -1 or self.repeat[idx] == 2:
                # Running variance
                delta = self.data_hdu.data - mean
                mean += delta / count
                delta2 = self.data_hdu.data - mean
                M2 += delta * delta2

        # Standard deviation
        self.err_hdu = fits.PrimaryHDU(
            header=self.data_hdu.header, data=(jnp.sqrt(M2 / (count - 1)))
        )

        self.save_error()
        return None
