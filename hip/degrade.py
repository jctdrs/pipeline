import typing

from util import read

from scipy.ndimage import zoom
import jax.numpy as jnp  # noqa

import astropy
from astropy.convolution import convolve_fft

import numpy as np
import numpy.ma as ma


class DegradeSingleton:
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


class Degrade(DegradeSingleton):
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
            return DegradeSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return DegradeMonteCarlo(*args, **kwargs)
        elif mode == "Automatic Differentiation":
            return DegradeAutomaticDifferentiation(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
    ]:
        data_hdu_invalid = ma.masked_invalid(self.data_hdu.data)
        self.load_kernel()
        self.scale_kernel(self.data_hdu)
        self.convert_from_Jyperpx_to_radiance(self.data_hdu)
        self.kernel_hdu.data /= np.sum(self.kernel_hdu.data)
        self.data_hdu.data = self.convolve(self.data_hdu.data, self.kernel_hdu.data)
        self.convert_from_radiance_to_Jyperpx(self.data_hdu)
        self.data_hdu.data[data_hdu_invalid.mask] = np.nan
        return self.data_hdu, self.err_hdu

    def load_kernel(self) -> None:
        hdu_kernel: astropy.io.image.PrimaryHDU = astropy.io.fits.open(
            self.task.parameters.kernel
        )
        self.kernel_hdu = hdu_kernel[0]
        return None

    def scale_kernel(self, hdu) -> None:
        pixel_data = read.pixel_size_arcsec(hdu.header)
        pixel_kernel = read.pixel_size_arcsec(self.kernel_hdu.header)
        kernel_xsize, kernel_ysize = read.shape(self.kernel_hdu.header)

        # Resize kernel if necessary
        if pixel_kernel != pixel_data:
            ratio = pixel_kernel / pixel_data
            size = ratio * kernel_xsize
            # Ensure an odd kernel
            if round(size) % 2 == 0:
                size += 1
                ratio = size / kernel_xsize

            self.kernel_hdu.data = zoom(self.kernel_hdu.data, ratio) / ratio**2
        return None

    def convolve(self, data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return convolve_fft(
            data,
            kernel,
            nan_treatment="interpolate",
            normalize_kernel=False,
            preserve_nan=True,
            fft_pad=True,
            boundary="fill",
            fill_value=0.0,
            allow_huge=True,
        )

    def convert_from_Jyperpx_to_radiance(self, hdu) -> None:
        pixel_size = read.pixel_size_arcsec(hdu.header)
        px_x: float = pixel_size * 2 * np.pi / 360
        px_y: float = pixel_size * 2 * np.pi / 360

        conversion_factor = (
            1e-26
            * pow((self.data.geometry.distance * 3.086e22), 2)
            * 4
            * np.pi
            / (px_x * px_y * 3.846e26)
        )

        hdu.data *= conversion_factor
        return

    def convert_from_radiance_to_Jyperpx(self, hdu) -> None:
        pixel_size = read.pixel_size_arcsec(hdu.header)
        px_x: float = pixel_size * 2 * np.pi / 360
        px_y: float = pixel_size * 2 * np.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.data.geometry.distance * 3.086e22), 2) * 4 * np.pi
        )

        hdu.data *= conversion_factor
        return


class DegradeSinglePass(Degrade):
    pass


class DegradeMonteCarlo(Degrade):
    pass


class DegradeAutomaticDifferentiation(Degrade):
    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
    ]:
        self.load_kernel()
        self.scale_kernel(self.err_hdu)
        self.kernel_hdu.data /= np.sum(self.kernel_hdu.data)

        self.convert_from_Jyperpx_to_radiance(self.err_hdu)
        self.err_hdu.data = self.convolve(self.err_hdu.data**2, self.kernel_hdu.data**2)

        self.err_hdu.data = jnp.sqrt(self.err_hdu.data)
        self.convert_from_radiance_to_Jyperpx(self.err_hdu)

        return self.data_hdu, self.err_hdu
