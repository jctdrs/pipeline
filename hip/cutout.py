import typing

import numpy as np

import astropy
import astropy.units as au
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord, ICRS


class Cutout:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        geom: dict,
        instruments: dict,
        use_jax: bool,
        raTrim: float,
        decTrim: float,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.use_jax = use_jax
        self.ra_trim = raTrim
        self.dec_trim = decTrim

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU, astropy.io.fits.hdu.image.PrimaryHDU, typing.Union[np.ndarray, typing.Any]
    ]:
        wcs = WCS(self.data_hdu.header)
        pos_center = SkyCoord(ra=self.geom["ra"] * au.deg, dec=self.geom["dec"] * au.deg, frame=ICRS)
        sizeTrim = (self.dec_trim * au.arcmin, self.ra_trim * au.arcmin)

        data_cutout = Cutout2D(self.data_hdu.data, position=pos_center, size=sizeTrim, wcs=wcs)
        self.data_hdu.data = data_cutout.data
        self.data_hdu.header.update(data_cutout.wcs.to_header())

        err_cutout = Cutout2D(self.err_hdu.data, position=pos_center, size=sizeTrim, wcs=wcs)
        self.err_hdu.data = err_cutout.data
        self.err_hdu.header.update(err_cutout.wcs.to_header())

        return self.data_hdu, self.err_hdu, None
