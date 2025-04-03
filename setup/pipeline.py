import itertools
import yaml
import copy
from dataclasses import dataclass
from dataclasses import field

from typing import Union
from typing import Any

from setup.pipeline_validation import PipelineStepUnrolled
from setup.pipeline_validation import Pipeline
from setup.bands_validation import Band
from setup.spec_validation import Specification

from hip import degrade
from hip import sky_subtract
from hip import regrid
from hip import foreground_mask
from hip import cutout
from hip import integrate
from hip import rms
from hip import test

from util import read

from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS

from reproject import reproject_interp

import numpy as np


INSTRUMENTS_CONFIG: str = "/home/jtedros/Repo/pipeline/data/config/instruments.yml"

Interface: dict[str, Any] = {
    "hip.degrade": degrade.Degrade.create,
    "hip.skySubtract": sky_subtract.SkySubtract.create,
    "hip.regrid": regrid.Regrid.create,
    "hip.cutout": cutout.Cutout.create,
    "hip.integrate": integrate.Integrate.create,
    "hip.foregroundMask": foreground_mask.ForegroundMask.create,
    "hip.rms": rms.Rms.create,
    "hip.test": test.Test,
}


@dataclass
class TaskControl:
    tasks: list[PipelineStepUnrolled]
    mode: str
    idx: int

    # Only used for MonteCarloPipeline
    MC_diagnosis: list = field(default_factory=list)
    repeat: list = field(default_factory=list)


class PipelineGeneric:
    def __init__(self, spec):
        self.spec = spec
        self._load_instruments()

    @classmethod
    def create(
        cls, spec: Pipeline
    ) -> Union["AnalyticPipeline", "MonteCarloPipeline", "SinglePassPipeline"]:
        if spec.config.mode == "Single Pass":
            return SinglePassPipeline(spec)

        elif spec.config.mode == "Analytic":
            SinglePassPipeline(spec).execute()
            return AnalyticPipeline(spec)

        elif spec.config.mode == "Monte-Carlo":
            SinglePassPipeline(spec).execute()
            return MonteCarloPipeline(spec)

        else:
            msg = f"[ERROR] Mode '{spec.config.mode}' not recognized"
            raise ValueError(msg)

    def load_data(self, band: Band) -> None:
        inp_path = band.input
        hdul = fits.open(inp_path)
        self.data_hdu = hdul[0]

        unit = read.unit(self.data_hdu.header)
        if "mJy/beam" in unit and "NIKA2" in band.name:
            beam_deg = read.BMAJ(self.data_hdu.header)
            px_size_deg = read.pixel_size_arcsec(self.data_hdu.header) / 3600

            conversion_factor = (
                px_size_deg**2 / (np.pi * beam_deg**2 / (4 * 0.693))
            ) * 1e-3
            self.data_hdu.data *= conversion_factor
        elif "Jy/px" in unit or "Jy/pix" in unit:
            pass
        else:
            msg = f"[ERROR] Unit should be Jy/px except for NIKA maps. Input {unit}."
            raise ValueError(msg)
        return None

    def load_error(self, band: Band) -> None:
        err_path = band.error
        if err_path is None:
            self.err_hdu = None
        else:
            hdul = fits.open(err_path)
            self.err_hdu = hdul[0]

        unit = read.unit(self.data_hdu.header)
        if "mJy/beam" in unit and "NIKA2" in band.name:
            beam_deg = read.BMAJ(self.err_hdu.header)
            px_size_deg = read.pixel_size_arcsec(self.err_hdu.header) / 3600

            conversion_factor = (
                px_size_deg**2 / (np.pi * beam_deg**2 / (4 * 0.693))
            ) * 1e-3
            self.err_hdu.data *= conversion_factor
        elif "Jy/px" in unit or "Jy/pix" in unit:
            pass
        else:
            msg = f"[ERROR] Unit should be Jy/px except for NIKA maps. Input {unit}."
            raise ValueError(msg)
        return None

    def save_data(self, band: Band) -> None:
        name = band.name
        out_path = band.output

        fits.writeto(
            f"{out_path}/{name}.fits",
            self.data_hdu.data,
            self.data_hdu.header,
            overwrite=True,
        )
        return None

    def save_error(self, band: Band, mode: str) -> None:
        name: str = band.name
        out_path: str = band.output
        if self.err_hdu is not None:
            fits.writeto(
                f"{out_path}/{name}_Error_{mode}.fits",
                self.err_hdu.data,
                self.err_hdu.header,
                overwrite=True,
            )
        return None

    def execute(self):
        return None

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
        return None

    def _set_task_control(self, band: Band) -> None:
        return None

    def _get_before_tasks_for_band(
        self,
        band: Band,
    ) -> list[PipelineStepUnrolled]:
        return [
            PipelineStepUnrolled(
                step=task.step,
                diagnosis=task.diagnosis,
                parameters=params,
            )
            for task in self.spec.before
            for params in task.parameters
            if params.band in {band.name, "all"}
        ]

    def _get_pipeline_tasks_for_band(self, band: Band) -> list[PipelineStepUnrolled]:
        return [
            PipelineStepUnrolled(
                step=task.step,
                diagnosis=task.diagnosis,
                parameters=params,
            )
            for task in self.spec.pipeline
            for params in task.parameters
            if params.band in {band.name, "all"}
        ]


class AnalyticPipeline(PipelineGeneric):
    def __init__(self, spec: Specification):
        super().__init__(spec)
        print("[INFO] Starting Analytic Pipeline")

    def _set_task_control(self, band: Band) -> None:
        unrolled_before_tasks = self._get_before_tasks_for_band(band)
        unrolled_pipeline_tasks = self._get_pipeline_tasks_for_band(band)
        self.task_control = TaskControl(
            mode="Analytic",
            tasks=unrolled_before_tasks + unrolled_pipeline_tasks,
            idx=0,
        )
        return None

    def execute(self) -> None:
        bands = self.spec.data.bands
        for band in bands:
            self.load_data(band)
            self.load_error(band)
            self._set_task_control(band)
            if self.err_hdu is None:
                std_data = mad_std(self.data_hdu.data, ignore_nan=True)
                self.err_hdu = fits.PrimaryHDU(
                    header=fits.Header(),
                    data=np.full_like(self.data_hdu.data, std_data),
                )[0]

            for idx, task in enumerate(self.task_control.tasks):
                self.task_control.idx = idx
                self.data_hdu, self.err_hdu = Interface[task.step](
                    task_control=self.task_control,
                    data_hdu=self.data_hdu,
                    err_hdu=self.err_hdu,
                    data=self.spec.data,
                    task=task,
                    band=band,
                    instruments=self.instruments,
                ).run()

            self.save_error(band, "AD")

        return None


class SinglePassPipeline(PipelineGeneric):
    def __init__(self, spec: Specification):
        super().__init__(spec)
        print("[INFO] Starting Single Pass Pipeline")

    def _set_task_control(self, band: Band) -> None:
        unrolled_before_tasks = self._get_before_tasks_for_band(band)
        unrolled_pipeline_tasks = self._get_pipeline_tasks_for_band(band)

        self.task_control = TaskControl(
            idx=0,
            mode="Single Pass",
            tasks=unrolled_before_tasks + unrolled_pipeline_tasks,
        )
        return None

    def execute(self) -> None:
        bands = self.spec.data.bands
        for band in bands:
            self.load_data(band)
            self._set_task_control(band)

            for idx, task in enumerate(self.task_control.tasks):
                self.data_hdu, _ = Interface[task.step](
                    task_control=self.task_control,
                    data_hdu=self.data_hdu,
                    err_hdu=None,
                    data=self.spec.data,
                    task=task,
                    band=band,
                    instruments=self.instruments,
                ).run()

            self.save_data(band)
        return None


class MonteCarloPipeline(PipelineGeneric):
    def __init__(self, spec: Specification):
        super().__init__(spec)
        print("[INFO] Starting Monte-Carlo Pipeline")

    def _set_task_control(self, band: Band) -> None:
        niter: int = self.spec.config.niter
        MC_diagnosis: list[bool] = []
        repeat: list[int] = []
        tasks: list[PipelineStepUnrolled] = []

        unrolled_before_tasks = self._get_before_tasks_for_band(band)
        unrolled_pipeline_tasks = self._get_pipeline_tasks_for_band(band)

        if unrolled_before_tasks:
            tasks.extend(unrolled_before_tasks)
            repeat.extend([0] * len(unrolled_before_tasks))
            MC_diagnosis.extend([False] * len(unrolled_before_tasks))

        tasks.extend(
            itertools.chain.from_iterable(
                itertools.repeat(unrolled_pipeline_tasks, niter)
            )
        )
        if len(unrolled_pipeline_tasks) == 1:
            repeat.extend([2] * niter)
            MC_diagnosis.extend([False] * (niter - 1) + [True])
        else:
            repeat.extend(
                ([1] + [0] * (len(unrolled_pipeline_tasks) - 2) + [-1]) * niter
            )
            MC_diagnosis.extend(
                [False] * len(self.spec.pipeline) * (niter - 1)
                + [True] * len(self.spec.pipeline)
            )

        self.task_control = TaskControl(
            mode="Monte-Carlo",
            tasks=tasks,
            repeat=repeat,
            MC_diagnosis=MC_diagnosis,
            idx=0,
        )
        return None

    def execute(self) -> None:
        bands: list = self.spec.data.bands

        for band in bands:
            count: float = 0
            mean: float = 0
            delta: float = 0
            delta2: float = 0
            M2: float = 0

            self.load_data(band)
            self.load_error(band)
            self._set_task_control(band)

            if self.err_hdu is None:
                std_data = mad_std(self.data_hdu.data, ignore_nan=True)
                self.err_hdu = fits.PrimaryHDU(
                    header=self.data_hdu.header,
                    data=np.full_like(self.data_hdu.data, std_data),
                )

            self.err_hdu.data, _ = reproject_interp(
                self.err_hdu,
                WCS(self.data_hdu.header),
                shape_out=self.data_hdu.data.shape,
            )

            for idx, task in enumerate(self.task_control.tasks):
                self.task_control.idx = idx
                if (
                    self.task_control.repeat[idx] == 1
                    or self.task_control.repeat[idx] == 2
                ):
                    if "original_data_hdu" not in locals():
                        original_data_hdu = self.data_hdu
                        original_err_hdu = self.err_hdu

                    self.data_hdu = fits.PrimaryHDU(
                        header=original_data_hdu.header,
                        data=np.array(original_data_hdu.data)
                        + np.array(original_err_hdu.data)
                        * np.random.normal(size=original_err_hdu.data.shape),
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

                if (
                    self.task_control.repeat[idx] == -1
                    or self.task_control.repeat[idx] == 2
                ):
                    # Running variance
                    print(
                        f"[INFO] Monte-Carlo iteration {count+1}/{self.spec.config.niter} \r",
                        flush=True,
                        end="",
                    )
                    count += 1
                    delta = self.data_hdu.data - mean
                    mean += delta / count
                    delta2 = self.data_hdu.data - mean
                    M2 += delta * delta2

            # Standard deviation
            self.err_hdu = fits.PrimaryHDU(
                header=self.data_hdu.header, data=(np.sqrt(M2 / (count - 1)))
            )

            del original_data_hdu
            self.save_error(band, "MC")
        return None
