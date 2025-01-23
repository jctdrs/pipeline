from typing import Optional

from util import read

import jax.numpy as jnp

import astropy
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel

from scipy.ndimage import zoom


class Degrade:
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
        mode = kwargs["task_control"].mode
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
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        self.load_kernel()
        self.kernel_hdu.data = jnp.array(self.kernel_hdu.data)
        self.kernel_hdu.data /= jnp.sum(self.kernel_hdu.data)
        self.convert_from_Jyperpx_to_radiance(self.data_hdu)
        self.data_hdu.data = self.convolve(self.data_hdu.data, self.kernel_hdu.data)
        self.convert_from_radiance_to_Jyperpx(self.data_hdu)

        return self.data_hdu, self.err_hdu

    def load_kernel(self) -> None:
        if self.task.parameters.kernel is not None:
            hdu_kernel: astropy.io.image.PrimaryHDU = astropy.io.fits.open(
                self.task.parameters.kernel
            )

            self.kernel_hdu = hdu_kernel[0]
            pixel_scale_i = read.pixel_size_arcsec(self.data_hdu.header)
            pixel_scale_k = read.pixel_size_arcsec(self.kernel_hdu.header)

            if pixel_scale_k != pixel_scale_i:
                ratio = pixel_scale_k / pixel_scale_i
                size = ratio * self.kernel_hdu.data.shape[0]
                if round(size) % 2 == 0:
                    size += 1
                    ratio = size / self.kernel_hdu.data.shape[0]

                self.kernel_hdu.data = zoom(self.kernel_hdu.data, ratio) / ratio**2

        elif self.task.parameters.target is not None:
            input_resolution = self.instruments[self.band.name]["resolutionArcsec"]
            target_resolution = self.task.parameters.target
            if target_resolution <= input_resolution:
                msg = "[ERROR] Cannot degrade to lower resolution."
                raise ValueError(msg)

            match_std = (target_resolution**2 - input_resolution**2) ** 0.5
            r_trim = jnp.sqrt(2 * match_std**2 * jnp.log(1 / (1 - 0.999)))
            grid_size = int(jnp.ceil(2 * r_trim)) | 1

            kernel = Gaussian2DKernel(
                match_std,
                x_size=grid_size,
                y_size=grid_size,
            )
            header = astropy.io.fits.Header()
            self.kernel_hdu = astropy.io.fits.PrimaryHDU(data=kernel, header=header)

        return None

    def convolve(self, data: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
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
        px_x: float = pixel_size * 2 * jnp.pi / 360
        px_y: float = pixel_size * 2 * jnp.pi / 360

        conversion_factor = (
            1e-26
            * pow((self.data.geometry.distance * 3.086e22), 2)
            * 4
            * jnp.pi
            / (px_x * px_y * 3.846e26)
        )

        hdu.data *= conversion_factor
        return None

    def convert_from_radiance_to_Jyperpx(self, hdu) -> None:
        pixel_size = read.pixel_size_arcsec(hdu.header)
        px_x: float = pixel_size * 2 * jnp.pi / 360
        px_y: float = pixel_size * 2 * jnp.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.data.geometry.distance * 3.086e22), 2) * 4 * jnp.pi
        )

        hdu.data *= conversion_factor
        return None


class DegradeSinglePass(Degrade):
    pass


class DegradeMonteCarlo(Degrade):
    pass


class DegradeAutomaticDifferentiation(Degrade):
    def run(
        self,
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        self.load_kernel()
        self.kernel_hdu.data /= jnp.sum(self.kernel_hdu.data)
        self.convert_from_Jyperpx_to_radiance(self.err_hdu)
        self.err_hdu.data = self.convolve(self.err_hdu.data**2, self.kernel_hdu.data**2)

        self.err_hdu.data = jnp.sqrt(self.err_hdu.data)
        self.convert_from_radiance_to_Jyperpx(self.err_hdu)

        return self.data_hdu, self.err_hdu
