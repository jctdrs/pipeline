from typing import Optional
from typing import Tuple

from packages import pipeline
from models import spec

import astropy

class Multiply:
    _instance = None
    _mode = None
    _band = None

    def __new__(cls, *args, **kwargs):
        mode = kwargs["task_control"].mode
        band = kwargs["band"].name
        if (
            cls._instance is None
            or (mode is None or mode != cls._mode)
            or (band is None or band != cls._band)
        ):
            cls._instance = super().__new__(cls)
            cls._mode = mode
            cls._band = band
        return cls._instance

    def __init__(
        self,
        task_control,
        data_hdu,
        err_hdu,
        data,
        task,
        band,
    ):
        self.task_control = task_control
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.data = data
        self.task = task
        self.band = band

    @classmethod
    def create(cls, *args, **kwargs):
        mode = kwargs["task_control"].mode
        if mode == "Single Pass":
            return SubtractSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return SubtractMonteCarlo(*args, **kwargs)
        elif mode == "Analytic":
            return SubtractAnalytic(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        if self.task.parameters.resultOf:
            spc = spec.Specification(self.task.parameters.resultOf).validate()
            pipe = pipeline.PipelineGeneric.create(spc)
            data_hdu, err_hdu = pipe.execute()

            self.data_hdu.data *= data_hdu.data

        elif self.task.parameters.target:
            self.load_target()
            self.data_hdu.data *= self.target_hdu.data

        elif self.task.parameters.factor:
            self.data_hdu.data *= self.task.parameters.factor

        return self.data_hdu, self.err_hdu

    def load_target(self):
        target_hdu: astropy.io.image.PrimaryHDU = astropy.io.fits.open(
            self.task.parameters.target
        )

        self.target_hdu = target_hdu[0]
        return None

class MultiplyMonteCarlo(Multiply):
    pass

class MultiplyAnalytic(Multiply):
    pass

class MultiplySinglePass(Multiply):
    pass
