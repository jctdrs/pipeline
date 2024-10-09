import math
import typing

import numpy as np
import numpy.ma as ma

import astropy
from astropy.wcs import WCS
from astropy.stats import SigmaClip

from photutils import background

import pyregion

from util import read


class Background:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        name: str,
        body: str,
        geom: dict,
        instruments: dict,
        use_jax: bool,
        cellSize: float,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.name = name
        self.body = body
        self.geom = geom
        self.instruments = instruments
        self.use_jax = use_jax
        self.cell_size = cellSize

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[np.ndarray, typing.Any],
    ]:
        mask = ma.masked_invalid(self.data_hdu.data).mask
        self.data_hdu.data[mask] = 0.0
        wcs = WCS(self.data_hdu.header)
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        pos = wcs.all_world2pix(self.geom["ra"], self.geom["dec"], 0)
        rma = math.ceil(self.geom["semiMajorAxis"] / 2 / pixel_size)
        rmi = math.ceil(
            self.geom["semiMajorAxis"] / 2 / self.geom["axialRatio"] / pixel_size
        )

        region = """
                image
                ellipse({},{},{},{},{})
                """.format(pos[0], pos[1], rma, rmi, self.geom["positionAngle"])

        r = pyregion.parse(region)

        bkg_mask = r.get_mask(hdu=self.data_hdu)
        xsize, ysize = read.shape(self.data_hdu.header)

        while True:
            try:
                lcell_arcsec = (
                    self.cell_size * self.instruments[self.name]["RESOLUTION"]["VALUE"]
                )
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
                interpolator = background.BkgZoomInterpolator(
                    order=3, mode="reflect", grid_mode=True
                )
                bkg_estimator = background.SExtractorBackground()
                bkgrms_estimator = background.StdBackgroundRMS()
                bkg = background.Background2D(
                    self.data_hdu.data,
                    (ncells1, ncells2),
                    edge_method="pad",
                    sigma_clip=sigma_clip,
                    interpolator=interpolator,
                    mask=bkg_mask,
                    exclude_percentile=10.0,
                    bkg_estimator=bkg_estimator,
                    bkgrms_estimator=bkgrms_estimator,
                )
                break
            except ValueError:
                self.cell_size += 0.5
                continue

        self.data_hdu.data = self.data_hdu.data - bkg.background
        self.data_hdu.data[mask] = np.nan

        return self.data_hdu, self.err_hdu, None
