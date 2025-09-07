import itertools
import copy
from dataclasses import dataclass
from dataclasses import field

from typing import Union
from typing import Any
from typing import List
from typing import Dict

from models.pipeline import PipelineStepUnrolled
from models.pipeline import Pipeline
from models.bands import Band

from packages.hip import degrade
from packages.hip import sky_subtract
from packages.hip import regrid
from packages.hip import foreground_mask
from packages.hip import cutout
from packages.hip import integrate
from packages.hip import test
from packages.core import subtract
from packages.core import multiply

from utilities import read

from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import WCS

from reproject import reproject_interp

import numpy as np


Interface: Dict[str, Any] = {
    "hip.degrade": degrade.Degrade.create,
    "hip.skySubtract": sky_subtract.SkySubtract.create,
    "hip.regrid": regrid.Regrid.create,
    "hip.cutout": cutout.Cutout.create,
    "hip.integrate": integrate.Integrate.create,
    "hip.foregroundMask": foreground_mask.ForegroundMask.create,
    "hip.test": test.Test.create,
    "core.subtract": subtract.Subtract,
    "core.multiply": multiply.Multiply,
}


@dataclass
class TaskControl:
    tasks: List[PipelineStepUnrolled]
    mode: str
    idx: int

    # Only used for MonteCarloPipeline
    MC_diagnosis: List = field(default_factory=list)
    repeat: List = field(default_factory=list)


class PipelineGeneric:
    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def create(
        cls, pipe: Pipeline
    ) -> Union["AnalyticPipeline", "MonteCarloPipeline", "SinglePassPipeline"]:
        if pipe.config.mode == "Single Pass":
            return SinglePassPipeline(pipe)

        elif pipe.config.mode == "Analytic":
            SinglePassPipeline(pipe).execute()
            return AnalyticPipeline(pipe)

        elif pipe.config.mode == "Monte-Carlo":
            SinglePassPipeline(pipe).execute()
            return MonteCarloPipeline(pipe)

        else:
            msg = f"[ERROR] Mode '{pipe.config.mode}' not recognized"
            raise ValueError(msg)

    def load_data(self, band: Band) -> None:
        inp_path = band.input
        hdul = fits.open(inp_path)
        self.data_hdu = hdul[0]

        unit = read.unit(self.data_hdu.header)
        if "mJy/beam" in unit and "NIKA2" in band.name:
            beam_deg = self.band.resolution / 3600
            px_size_deg = self.band.pixelSize / 3600

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

        if self.err_hdu is not None:
            unit = read.unit(self.data_hdu.header)
            if "mJy/beam" in unit and "NIKA2" in band.name:
                beam_deg = self.band.resolution / 3600
                px_size_deg = self.band.pixelSize / 3600

                conversion_factor = (
                    px_size_deg**2 / (np.pi * beam_deg**2 / (4 * 0.693))
                ) * 1e-3
                self.err_hdu.data *= conversion_factor
            elif "Jy/px" in unit or "Jy/pix" in unit:
                pass
            else:
                msg = (
                    f"[ERROR] Unit should be Jy/px except for NIKA maps. Input {unit}."
                )
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
        pass

    def _set_task_control(self, band: Band) -> None:
        pass

    def _get_before_tasks_for_band(
        self,
        band: Band,
    ) -> List[PipelineStepUnrolled]:
        return [
            PipelineStepUnrolled(
                step=task.step,
                diagnosis=task.diagnosis,
                parameters=params,
            )
            for task in self.pipe.before
            for params in task.parameters
            if params.band in {band.name, "all"}
        ]

    def _get_pipeline_tasks_for_band(self, band: Band) -> List[PipelineStepUnrolled]:
        return [
            PipelineStepUnrolled(
                step=task.step,
                diagnosis=task.diagnosis,
                parameters=params,
            )
            for task in self.pipe.pipeline
            for params in task.parameters
            if params.band in {band.name, "all"}
        ]


class AnalyticPipeline(PipelineGeneric):
    def __init__(self, pipe: Pipeline):
        super().__init__(pipe)
        print("[INFO] Starting Analytic Pipeline")
        print("[WARNING] Analytic method is still under testing")

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
        bands = self.pipe.data.bands
        for band in bands:
            self.load_data(band)
            self.load_error(band)
            self._set_task_control(band)
            if self.err_hdu is None:
                std_data = mad_std(self.data_hdu.data, ignore_nan=True)
                self.err_hdu = fits.PrimaryHDU(
                    header=self.data_hdu.header,
                    data=np.full_like(self.data_hdu.data, std_data),
                )

            self.err_hdu.header.set("ERRCORR", "False")

            for idx, task in enumerate(self.task_control.tasks):
                self.task_control.idx = idx
                self.data_hdu, self.err_hdu = Interface[task.step](
                    task_control=self.task_control,
                    data_hdu=self.data_hdu,
                    err_hdu=self.err_hdu,
                    data=self.pipe.data,
                    task=task,
                    band=band,
                ).run()

            self.save_error(band, "AD")

        return self.data_hdu, self.err_hdu


class SinglePassPipeline(PipelineGeneric):
    def __init__(self, pipe: Pipeline):
        super().__init__(pipe)
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
        bands = self.pipe.data.bands
        for band in bands:
            self.load_data(band)
            self._set_task_control(band)

            for idx, task in enumerate(self.task_control.tasks):
                self.data_hdu, _ = Interface[task.step](
                    task_control=self.task_control,
                    data_hdu=self.data_hdu,
                    err_hdu=None,
                    data=self.pipe.data,
                    task=task,
                    band=band,
                ).run()

            self.save_data(band)

        return self.data_hdu, None


class MonteCarloPipeline(PipelineGeneric):
    def __init__(self, pipe: Pipeline):
        super().__init__(pipe)
        print("[INFO] Starting Monte-Carlo Pipeline")

    def _set_task_control(self, band: Band) -> None:
        niter: int = self.pipe.config.niter
        MC_diagnosis: List[bool] = []
        repeat: List[int] = []
        tasks: List[PipelineStepUnrolled] = []

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
                [False] * len(self.pipe.pipeline) * (niter - 1)
                + [True] * len(self.pipe.pipeline)
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
        bands: List = self.pipe.data.bands

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

            self.err_hdu.header.set("ERRCORR", "False")

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
                    data=self.pipe.data,
                    task=task,
                    band=band,
                ).run()

                if (
                    self.task_control.repeat[idx] == -1
                    or self.task_control.repeat[idx] == 2
                ):
                    # Running variance
                    print(
                        f"[INFO] Monte-Carlo iteration {count+1}/{self.pipe.config.niter} \r",
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

        return self.data_hdu, self.err_hdu
