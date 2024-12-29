import itertools
import typing
import yaml
import copy

from setup import bands_validation

from astropy.io import fits
from astropy.stats import mad_std

import jax
import jax.numpy as jnp

from hip import degrade
from hip import sky_subtract
from hip import regrid
from hip import foreground_mask
from hip import cutout
from hip import integrate

from setup import spec_validation

from util import test

INSTRUMENTS_CONFIG = "/home/jtedros/Repo/pipeline/data/config/instruments.yml"

Interface: dict = {
    "hip.degrade": degrade.Degrade.create,
    "hip.skySubtract": sky_subtract.SkySubtract.create,
    "hip.regrid": regrid.Regrid.create,
    "hip.cutout": cutout.Cutout.create,
    "hip.integrate": integrate.Integrate.create,
    "hip.foregroundMask": foreground_mask.ForegroundMask.create,
}


class Pipeline:
    def __init__(self, spec):
        self.spec = spec
        self._load_instruments()

    @classmethod
    def create(
        cls, spec: spec_validation.Specification
    ) -> typing.Union[
        "DifferentialPipeline", "MonteCarloPipeline", "SinglePassPipeline"
    ]:
        if spec.config.mode == "Single Pass":
            return SinglePassPipeline(spec)

        elif spec.config.mode == "Automatic Differentiation":
            return DifferentialPipeline(spec)

        elif spec.config.mode == "Monte-Carlo":
            SinglePassPipeline(spec).execute()
            return MonteCarloPipeline(spec)

        else:
            msg = f"[ERROR] Mode '{spec.config.mode}' not recognized"
            raise ValueError(msg)

    def load_input(self, band: bands_validation.Band) -> None:
        inp_path: str = band.input
        hdul = fits.open(inp_path)
        self.data_hdu = hdul[0]
        return

    def load_error(self, band: bands_validation.Band) -> None:
        err_path: str = band.error
        if err_path is None:
            return

        hdul = fits.open(err_path)
        self.err_hdu = hdul[0]
        return

    def save_data(self, band: bands_validation.Band) -> None:
        name: str = band.name
        out_path: str = band.output

        fits.writeto(
            f"{out_path}/{name}.fits",
            self.data_hdu.data,
            self.data_hdu.header,
            overwrite=True,
        )
        return

    def save_error(self, band: bands_validation.Band) -> None:
        name: str = band.name
        out_path: str = band.output

        fits.writeto(
            f"{out_path}/{name}_Error.fits",
            self.err_hdu.data,
            self.err_hdu.header,
            overwrite=True,
        )
        return

    def execute(self):
        pass

    def _load_instruments(self) -> None:
        try:
            f = open(INSTRUMENTS_CONFIG, "r")
        except OSError:
            msg = "[ERROR] File 'config/instruments.yml' not found."
            raise OSError(msg)

        # Check if specification is valid YAML
        # In case of failure, capture line/col for debug
        try:
            self.instruments = yaml.load(f, yaml.SafeLoader)
            f.close()
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            line = e.problem_mark.line + 1  # type: ignore
            column = e.problem_mark.column + 1  # type: ignore
            msg = f"[ERROR] YAML parsing error at line {line}, column {column}."
            raise ValueError(msg)
        return

    def _set_tasks(self) -> None:
        pass


# TODO: If error not define in YAML then abort or add error from std
class DifferentialPipeline(Pipeline):
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)
        self._set_tasks()

    def _set_tasks(self) -> None:
        pass

    def execute(self) -> None:
        self.load_input()

        # TODO: bands is a list
        error: str = self.spec.data.bands.error

        if error is None:
            self.load_error()
        else:
            std_data = mad_std(self.data_hdu.data, ignore_nan=True)
            self.err_hdu = fits.PrimaryHDU(
                header=fits.Header(), data=jnp.full_like(self.data_hdu.data, std_data)
            )[0]

        first_step_with_grad: bool = True
        pipeline_grad = None

        body: str = self.spec.data.body
        # TODO: bands is a list
        name: str = self.spec.data.bands.name
        out_path: str = self.spec.data.bands.output

        # Loop over steps in pipeline
        for task in self.file_mng.tasks:
            self.data_hdu, self.err_hdu, grad_arr = Interface[task["step"]](
                self.data_hdu,
                self.err_hdu,
                out_path,
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
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)
        print("[INFO] Starting Single Pass Pipeline")
        self._set_tasks()

    def _set_tasks(self) -> None:
        self.tasks: list = [task for task in self.spec.pipeline]
        return

    def execute(self) -> None:
        bands: list = self.spec.data.bands
        for band in bands:
            # TODO: These need to be like each other
            self.load_input(band)
            self.err_hdu = None

            for task in self.tasks:
                self.data_hdu, self.err_hdu, _ = Interface[task.step](
                    mode="Single Pass",
                    data_hdu=self.data_hdu,
                    err_hdu=self.err_hdu,
                    data=self.spec.data,
                    task=task,
                    band=band,
                    instruments=self.instruments,
                ).run()

            self.save_data(band)
        return None


class MonteCarloPipeline(Pipeline):
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)
        print("[INFO] Starting Monte-Carlo Pipeline")
        self._set_tasks()

    def _set_tasks(self) -> None:
        self.tasks: list = []
        self.repeat: list = []

        niter = self.spec.config.niter
        self.tasks.extend(
            itertools.chain.from_iterable(itertools.repeat(self.spec.pipeline, niter))
        )
        if len(self.spec.pipeline) == 1:
            self.repeat.extend([2] * niter)
        else:
            self.repeat.extend(
                ([1] + [0] * (len(self.spec.pipeline) - 2) + [-1]) * niter
            )
        return

    def execute(self) -> None:
        key = jax.random.key(638)
        bands: list = self.spec.data.bands

        for band in bands:
            count: float = 0
            mean: float = 0
            M2: float = 0

            self.load_input(band)
            err_path = band.error
            if err_path is not None:
                self.load_error(band)
            else:
                std_data = mad_std(self.data_hdu.data, ignore_nan=True)
                self.err_hdu = fits.PrimaryHDU(
                    header=fits.Header(),
                    data=jnp.full_like(self.data_hdu.data, std_data),
                )[0]

            for idx, task in enumerate(self.tasks):
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

                self.data_hdu, self.err_hdu, _ = Interface[task.step](
                    mode="Monte-Carlo",
                    data_hdu=self.data_hdu,
                    err_hdu=self.err_hdu,
                    data=self.spec.data,
                    task=task,
                    band=band,
                    instruments=self.instruments,
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

            self.save_error(band)
        return None
