import astropy
from astropy.wcs import WCS
from astropy.io import fits
import typing
from reproject import reproject_interp

from util import read


class Reproject:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any],
        geom: dict,
        instruments: dict,
        target: str,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.target = target

    def run(self):
        with fits.open(self.target) as hdul:
            hdr_target = hdul[0].header
            xsize, ysize = read.shape(hdr_target)

        wcs = WCS(hdr_target)
        self.data_hdu.data, _ = reproject_interp(self.data_hdu, wcs, shape_out=(xsize, ysize))

        self.data_hdu.header.update(wcs.to_header())

        return self.data_hdu, self.err_hdu
