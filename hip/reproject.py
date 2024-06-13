import math
import typing

import numpy as np

import astropy

from astropy.wcs import WCS
from astropy.io import fits

from reproject import reproject_interp

from util import read


class Reproject:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        geom: dict,
        instruments: dict,
        use_jax: bool,
        target: str,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.use_jax = use_jax
        self.target = target

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU, astropy.io.fits.hdu.image.PrimaryHDU, typing.Union[np.ndarray, typing.Any]
    ]:
        self.convert_from_Jyperpx_to_radiance()
        self.reproject()
        self.convert_from_radiance_to_Jyperpx()
        return self.data_hdu, self.err_hdu, None

    def convert_from_Jyperpx_to_radiance(self) -> typing.Any:
        pixel_size = read.pixel_size(self.data_hdu.header)
        px_x: float = pixel_size[0] * 2 * math.pi / 360
        px_y: float = pixel_size[1] * 2 * math.pi / 360

        # divide by 3.846x10^26 (Lsun in Watt) to convert W/Hz/m2/sr in
        # Lsun/Hz/m2/sr multiply by the galaxy distance in m2 to get Lsun/Hz/sr
        conversion_factor = 1e-26 * pow((self.geom["distance"] * 3.086e22), 2) * 4 * math.pi / (px_x * px_y * 3.846e26)

        self.data_hdu.data *= conversion_factor
        return None

    def reproject(self) -> typing.Any:
        with fits.open(self.target) as hdul:
            hdr_target = hdul[0].header
            xsize, ysize = read.shape(hdr_target)

        wcs = WCS(hdr_target)
        self.data_hdu.data, _ = reproject_interp(input_data=self.data_hdu, output_projection=wcs)
        self.data_hdu.header.update(wcs.to_header())
        return None

    def convert_from_radiance_to_Jyperpx(self) -> typing.Any:
        pixel_size = read.pixel_size(self.data_hdu.header)
        px_x: float = pixel_size[0] * 2 * math.pi / 360
        px_y: float = pixel_size[1] * 2 * math.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.geom["distance"] * 3.086e22), 2) * 4 * math.pi
        )

        self.data_hdu.data *= conversion_factor
        return None
