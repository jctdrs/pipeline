import itertools
import typing
import yaml
import copy

from astropy.io import fits
from astropy.stats import mad_std

import jax
import jax.numpy as jnp

from hip import convolution
from hip import skysubtract
from hip import reproject
from hip import foreground
from hip import cutout
from hip import integrate

from setup import spec_validation

from util import test

INSTRUMENTS_CONFIG = "/home/jtedros/Repo/pipeline/data/config/instruments.yml"

Interface: dict = {
    "hip.convolution": convolution.Convolution,
    "hip.skySubtract": skysubtract.SkySubtract,
    "hip.reproject": reproject.Reproject,
    "hip.cutout": cutout.Cutout,
    "hip.integrate": integrate.Integrate,
    "hip.foreground": foreground.Foreground,
    "util.test": test.Test,
}


class Pipeline:
    def __init__(self, spec):
        self.spec = spec
        self._load_instruments()
        self._set_tasks()

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

    def load_input(self, idx: int) -> None:
        inp_path: str = self.spec.data.bands[idx].input
        hdul = fits.open(inp_path)
        self.data_hdu = hdul[0]
        return

    def load_error(self, idx: int) -> None:
        err_path: str = self.spec.data.bands[idx].error
        if err_path is None:
            return

        hdul = fits.open(err_path)
        self.err_hdu = hdul[0]
        return

    def save_data(self, idx) -> None:
        name: str = self.spec.data.bands[idx].name
        out_path: str = self.spec.data.bands[idx].output

        fits.writeto(
            f"{out_path}/{name}.fits",
            self.data_hdu.data,
            self.data_hdu.header,
            overwrite=True,
        )

    def save_error(self, idx) -> None:
        name: str = self.spec.data.bands[idx].name
        out_path: str = self.spec.data.bands[idx].output

        if self.err_hdu is not None:
            fits.writeto(
                f"{out_path}/{name}_Error.fits",
                self.err_hdu.data,
                self.err_hdu.header,
                overwrite=True,
            )

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

    def _set_tasks(self) -> None:
        self.single: list = []
        self.repeat: list = []
        self.tasks: list = []
        self.MC_diagnosis: list = []

        # Check for pipeline steps
        for item in self.spec.pipeline:
            self.single.append(item)

        niter = self.spec.config.niter
        if niter > 1:
            self.tasks.extend(
                itertools.chain.from_iterable(
                    itertools.repeat(self.spec.pipeline, niter)
                )
            )
            if len(self.spec.pipeline) == 1:
                self.repeat.extend([2] * niter)
                self.MC_diagnosis.extend([False] * (niter - 1))
                self.MC_diagnosis.extend([True])
            else:
                self.repeat.extend(
                    ([1] + [0] * (len(self.spec.pipeline) - 2) + [-1]) * niter
                )
                self.MC_diagnosis.extend(
                    [False] * len(self.spec.pipeline) * (niter - 1)
                )
                self.MC_diagnosis.extend([True] * len(self.spec.pipeline))
        else:
            self.tasks.extend(self.spec.pipeline)
        return


# TODO: If error not define in YAML then abort or add error from std
class DifferentialPipeline(Pipeline):
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)

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

    def execute(self) -> None:
        bands: list = self.spec.data.bands
        for idx, band in enumerate(bands):
            # TODO: These need to be like each other
            self.load_input(idx)
            self.err_hdu = None

            for task in self.single:
                self.data_hdu, self.err_hdu, _ = Interface[task.step](
                    self.data_hdu,
                    self.err_hdu,
                    self.spec.data,
                    task,
                    idx,
                    self.instruments,
                    MC_diagnosis=False,
                    differentiate=False,
                ).run()

            self.save_data(idx)
        return None


class MonteCarloPipeline(Pipeline):
    def __init__(self, spec: spec_validation.Specification):
        super().__init__(spec)

    def execute(self) -> None:
        key = jax.random.key(638)
        bands: list = self.spec.data.bands
        for jdx, band in enumerate(bands):
            count: float = 0
            mean: float = 0
            M2: float = 0

            MC_diagnosis: bool = False

            self.load_input(jdx)
            err_path = band.error
            if err_path is not None:
                self.load_error(jdx)
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
                    self.data_hdu,
                    self.err_hdu,
                    self.spec.data,
                    task,
                    jdx,
                    self.instruments,
                    MC_diagnosis=False,
                    differentiate=False,
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

            self.save_error(jdx)
        return None
