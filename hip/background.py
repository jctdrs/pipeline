import typing
import math

import astropy
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from photutils import background
import pyregion
import numpy.ma as ma

from util import read


class Background:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any],
        geom: dict,
        instruments: dict,
        cellSize: float,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.cell_size = cellSize

    def run(self):
        wcs = WCS(self.data_hdu.header)
        pixel_size = read.pixel_scale(self.data_hdu.header)
        pos = wcs.all_world2pix(self.geom["ra"], self.geom["dec"], 0)
        rma = math.ceil(self.geom["majorAxis"] / pixel_size)
        rmi = math.ceil(self.geom["majorAxis"] / self.geom["axialRatio"] / pixel_size)

        region = """
                image
                ellipse({},{},{},{},{})
                """.format(
            pos[0], pos[1], rma, rmi, self.geom["positionAngle"]
        )
        r = pyregion.parse(region)

        bkg_mask = r.get_mask(hdu=self.data_hdu)
        data_nan_masked = ma.masked_invalid(self.data_hdu.data)
        nan_mask = data_nan_masked.mask
        
        # TODO: Get the band_name. Maybe pass file_mng
        band_name = "NIKA2_1"
        xsize, ysize = read.shape(self.data_hdu.header)
        while True:
            try:
                lcell_arcsec = self.cell_size * self.instruments[band_name]["RESOLUTION"]["VALUE"]
                lcell_px = math.ceil(lcell_arcsec / pixel_size)
                ncells1 = math.ceil(xsize / lcell_px)
                ncells2 = math.ceil(ysize / lcell_px)

                sigma_clip = SigmaClip(
                    sigma=3.0,
                    maxiters=None,
                    cenfunc="median",
                    stdfunc="std",
                    grow=False,
                )
                interpolator = background.BkgZoomInterpolator(order=3, mode="reflect", grid_mode=True)
                bkg_estimator = background.SExtractorBackground()
                bkgrms_estimator = background.StdBackgroundRMS()
                bkg = background.Background2D(
                    self.data_hdu.data,
                    (ncells1, ncells2),
                    edge_method="pad",
                    sigma_clip=sigma_clip,
                    interpolator=interpolator,
                    mask=bkg_mask,
                    coverage_mask=nan_mask,
                    exclude_percentile=10.0,
                    bkg_estimator=bkg_estimator,
                    bkgrms_estimator=bkgrms_estimator,
                )
                break
            except ValueError:
                self.cell_size += 0.5
                continue

        self.data_hdu.data = self.data_hdu.data - bkg.background

        return self.data_hdu, self.err_hdu
