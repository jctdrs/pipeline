import itertools
import typing
import yaml
import copy

from setup import bands_validation
from setup import spec_validation

from hip import degrade
from hip import sky_subtract
from hip import regrid
from hip import foreground_mask
from hip import cutout
from hip import integrate
from hip import test

from astropy.io import fits
from astropy.stats import mad_std

import jax
import jax.numpy as jnp


INSTRUMENTS_CONFIG = "/home/jtedros/Repo/pipeline/data/config/instruments.yml"

Interface: dict = {
    "hip.degrade": degrade.Degrade.create,
    "hip.skySubtract": sky_subtract.SkySubtract.create,
    "hip.regrid": regrid.Regrid.create,
    "hip.cutout": cutout.Cutout.create,
    "hip.integrate": integrate.Integrate.create,
    "hip.foregroundMask": foreground_mask.ForegroundMask.create,
    "hip.test": test.Test,
}


class Pipeline:
    def __init__(self, spec):
        self.spec = spec
        self._load_instruments()

    @classmethod
    def create(
        cls, spec: spec_validation.Specification
    ) -> typing.Union[
        "AutomaticDifferentiationPipeline", "MonteCarloPipeline", "SinglePassPipeline"
    ]:
        if spec.config.mode == "Single Pass":
            return SinglePassPipeline(spec)

        elif spec.config.mode == "Automatic Differentiation":
            SinglePassPipeline(spec).execute()
            return AutomaticDifferentiationPipeline(spec)

        elif spec.config.mode == "Monte-Carlo":
            SinglePassPipeline(spec).execute()
            return MonteCarloPipeline(spec)

        else:
            msg = f"[ERROR] Mode '{spec.config.mode}' not recognized"
            raise ValueError(msg)

    def load_data(self, band: bands_validation.Band) -> None:
        inp_path: str = band.input
        hdul = fits.open(inp_path)
        self.data_hdu = hdul[0]

    def load_error(self, band: bands_validation.Band) -> None:
        err_path: str = band.error
        if err_path is None:
            self.err_hdu = None
        else:
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
        return

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

    def _set_task_control(self) -> None:
        return


class AutomaticDifferentiationPipeline(Pipeline):
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)
        print("[INFO] Starting Automatic Differentiation Pipeline")
        self._set_task_control()

    def _set_task_control(self) -> None:
        self.task_control = {
            "mode": "Automatic Differentiation",
            "tasks": [task for task in self.spec.pipeline],
            "idx": 0,
        }
        return

    def execute(self) -> None:
        bands: list = self.spec.data.bands
        for band in bands:
            self.load_data(band)
            self.load_error(band)
            if self.err_hdu is None:
                std_data = mad_std(self.data_hdu.data, ignore_nan=True)
                self.err_hdu = fits.PrimaryHDU(
                    header=fits.Header(),
                    data=jnp.full_like(self.data_hdu.data, std_data),
                )[0]

            for idx, task in enumerate(self.task_control["tasks"]):
                self.task_control["idx"] = idx
                self.data_hdu, self.err_hdu = Interface[task.step](
                    task_control=self.task_control,
                    data_hdu=self.data_hdu,
                    err_hdu=self.err_hdu,
                    data=self.spec.data,
                    task=task,
                    band=band,
                    instruments=self.instruments,
                ).run()

            self.err_hdu.data = jnp.sqrt(self.err_hdu.data)

            self.save_error(band)

        return


class SinglePassPipeline(Pipeline):
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)
        print("[INFO] Starting Single Pass Pipeline")
        self._set_task_control()

    def _set_task_control(self) -> None:
        self.task_control = {
            "mode": "Single Pass",
            "tasks": [task for task in self.spec.pipeline],
            "idx": 0,
        }
        return

    def execute(self) -> None:
        bands: list = self.spec.data.bands
        for band in bands:
            self.load_data(band)
            self.err_hdu = None

            for idx, task in enumerate(self.task_control["tasks"]):
                self.task_control["idx"] = idx
                self.data_hdu, self.err_hdu = Interface[task.step](
                    task_control=self.task_control,
                    data_hdu=self.data_hdu,
                    err_hdu=self.err_hdu,
                    data=self.spec.data,
                    task=task,
                    band=band,
                    instruments=self.instruments,
                ).run()

            self.save_data(band)
        return


class MonteCarloPipeline(Pipeline):
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)
        print("[INFO] Starting Monte-Carlo Pipeline")
        self._set_task_control()

    def _set_task_control(self) -> None:
        niter: int = self.spec.config.niter
        MC_diagnosis: list = []
        repeat: list = []
        tasks: list = []

        tasks.extend(
            itertools.chain.from_iterable(itertools.repeat(self.spec.pipeline, niter))
        )
        if len(self.spec.pipeline) == 1:
            repeat.extend([2] * niter)
            MC_diagnosis.extend([False] * (niter - 1) + [True])
        else:
            repeat.extend(([1] + [0] * (len(self.spec.pipeline) - 2) + [-1]) * niter)
            MC_diagnosis.extend(
                [False] * len(self.spec.pipeline) * (niter - 1)
                + [True] * len(self.spec.pipeline)
            )

        self.task_control = {
            "mode": "Monte-Carlo",
            "tasks": tasks,
            "repeat": repeat,
            "MC_diagnosis": MC_diagnosis,
            "idx": 0,
        }
        return

    def execute(self) -> None:
        key = jax.random.key(638)
        bands: list = self.spec.data.bands

        for band in bands:
            count: float = 0
            mean: float = 0
            M2: float = 0

            self.load_data(band)
            self.load_error(band)
            if self.err_hdu is None:
                std_data = mad_std(self.data_hdu.data, ignore_nan=True)
                self.err_hdu = fits.PrimaryHDU(
                    header=fits.Header(),
                    data=jnp.full_like(self.data_hdu.data, std_data),
                )[0]

            for idx, task in enumerate(self.task_control["tasks"]):
                self.task_control["idx"] = idx
                if (
                    self.task_control["repeat"][idx] == 1
                    or self.task_control["repeat"][idx] == 2
                ):
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

                self.data_hdu, self.err_hdu = Interface[task.step](
                    task_control=self.task_control,
                    data_hdu=self.data_hdu,
                    err_hdu=self.err_hdu,
                    data=self.spec.data,
                    task=task,
                    band=band,
                    instruments=self.instruments,
                ).run()

                # Running sum
                if (
                    self.task_control["repeat"][idx] == -1
                    or self.task_control["repeat"][idx] == 2
                ):
                    # Running variance
                    count += 1
                    delta = self.data_hdu.data - mean
                    mean += delta / count
                    delta2 = self.data_hdu.data - mean
                    M2 += delta * delta2

            # Standard deviation
            self.err_hdu = fits.PrimaryHDU(
                header=self.data_hdu.header, data=(jnp.sqrt(M2 / (count - 1)))
            )

            self.save_error(band)
        return
