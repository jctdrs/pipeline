import math
import typing

import jax.numpy as jnp

import astropy
from astropy.wcs import WCS
from astropy.io import fits

from reproject import reproject_interp

from util import read


class RegridSingleton:
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


class Regrid(RegridSingleton):
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
            return RegridSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return RegridMonteCarlo(*args, **kwargs)
        elif mode == "Automatic Differentiation":
            return RegridAutomaticDifferentiation(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.array, typing.Any],
    ]:
        self.convert_from_Jyperpx_to_radiance()
        with fits.open(self.task.parameters.target) as hdul:
            hdr_target = hdul[0].header

        wcs_out = WCS(hdr_target)
        self.data_hdu.data, _ = reproject_interp(
            input_data=self.data_hdu,
            output_projection=wcs_out,
        )
        self.data_hdu.header.update(wcs_out.to_header())
        self.convert_from_radiance_to_Jyperpx()

        return self.data_hdu, self.err_hdu, None

    def convert_from_Jyperpx_to_radiance(self) -> typing.Any:
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        px_x: float = pixel_size * 2 * math.pi / 360
        px_y: float = pixel_size * 2 * math.pi / 360

        # divide by 3.846x10^26 (Lsun in Watt) to convert W/Hz/m2/sr in
        # Lsun/Hz/m2/sr multiply by the galaxy distance in m2 to get Lsun/Hz/sr
        conversion_factor = (
            1e-26
            * pow((self.data.geometry.distance * 3.086e22), 2)
            * 4
            * math.pi
            / (px_x * px_y * 3.846e26)
        )

        self.data_hdu.data *= conversion_factor
        return None

    def convert_from_radiance_to_Jyperpx(self) -> typing.Any:
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        px_x: float = pixel_size * 2 * math.pi / 360
        px_y: float = pixel_size * 2 * math.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.data.geometry.distance * 3.086e22), 2) * 4 * math.pi
        )

        self.data_hdu.data *= conversion_factor
        return None


class RegridSinglePass(Regrid):
    pass


class RegridMonteCarlo(Regrid):
    pass


class RegridAutomaticDifferentiation(Regrid):
    pass
