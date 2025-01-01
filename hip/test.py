import typing

import astropy


class TestSingleton:
    _instance = None
    _mode = None

    def __new__(cls, *args, **kwargs):
        mode = kwargs["task_control"]["mode"]
        if cls._instance is None and (mode is None or mode != cls._mode):
            cls._instance = super().__new__(cls)
            cls._mode = mode
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class Test(TestSingleton):
    def __init__(
        self,
        task_control,
        data_hdu,
        err_hdu,
        data,
        task,
        band,
        instruments,
    ):
        self.task_control = task_control
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.data = data
        self.task = task
        self.band = band
        self.instruments = instruments

    @classmethod
    def create(cls, *args, **kwargs):
        mode = kwargs["task_control"]["mode"]
        if mode == "Single Pass":
            return TestSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return TestMonteCarlo(*args, **kwargs)
        elif mode == "Automatic Differentiation":
            return TestAutomaticDifferentiation(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
    ]:
        return self.data_hdu, self.err_hdu


class TestSinglePass(Test):
    pass


class TestMonteCarlo(Test):
    pass


class TestAutomaticDifferentiation(Test):
    pass
