import typing

from setup import pipeline_validation
from setup import data_validation

import astropy
from astropy.stats import SigmaClip
from astropy.wcs import WCS

from photutils import background

import jax.numpy as jnp

import pyregion

import copy

import numpy.ma as ma
import matplotlib.pyplot as plt

from util import read


class SkySubtractSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class SkySubtract(SkySubtractSingleton):
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        data: data_validation.Data,
        task: pipeline_validation.PipelineStep,
        idx: int,
        instruments: dict,
        MC_diagnosis: bool,
        differentiate: bool,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.task = task
        self.data = data
        self.band = self.data.bands[idx]
        self.instruments = instruments
        self.MC_diagnosis = MC_diagnosis
        self.differentiate = differentiate

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.array, typing.Any],
    ]:
        # This masking is needed to tame the Warning from photutils
        data_hdu_invalid = ma.masked_invalid(self.data_hdu.data)
        self.data_hdu.data[data_hdu_invalid.mask] = 0.0

        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        xsize, ysize = read.shape(self.data_hdu.header)
        lcell_px = jnp.ceil(
            self.task.parameters.cellSize
            * self.instruments[self.band.name]["RESOLUTION"]["VALUE"]
            / pixel_size
        )
        ncells1 = int(jnp.ceil(xsize / lcell_px))
        ncells2 = int(jnp.ceil(ysize / lcell_px))

        wcs = WCS(self.data_hdu.header)
        pos = wcs.all_world2pix(self.data.geometry.ra, self.data.geometry.dec, 0)
        rma = jnp.ceil(self.data.geometry.semiMajorAxis / 2 / pixel_size)
        rmi = jnp.ceil(
            self.data.geometry.semiMajorAxis
            / 2
            / self.data.geometry.axialRatio
            / pixel_size
        )

        region = pyregion.parse(
            """
                image
                ellipse({},{},{},{},{})
                """.format(pos[0], pos[1], rma, rmi, self.data.geometry.positionAngle)
        )

        bkg_mask = region.get_mask(hdu=self.data_hdu)

        bkg = background.Background2D(
            self.data_hdu.data,
            (ncells1, ncells2),
            edge_method="pad",
            sigma_clip=SigmaClip(
                sigma=3.0,
                maxiters=None,
                cenfunc="median",
                stdfunc="std",
                grow=False,
            ),
            interpolator=background.BkgZoomInterpolator(
                order=3, mode="reflect", grid_mode=True
            ),
            mask=bkg_mask,
            exclude_percentile=10.0,
            bkg_estimator=background.SExtractorBackground(),
            bkgrms_estimator=background.StdBackgroundRMS(),
        )

        self.data_hdu.data = self.data_hdu.data - bkg.background
        self.data_hdu.data[data_hdu_invalid.mask] = jnp.nan

        if self.task.diagnosis:
            mask_bkg = copy.copy(bkg.background)
            mask_bkg[data_hdu_invalid.mask] = jnp.nan

            sourcemask = copy.deepcopy(mask_bkg)
            sourcemask[bkg_mask] = jnp.nan

            plt.imshow(mask_bkg, origin="lower")
            plt.title(f"{self.data.body} {self.band.name} background map")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.yticks([])
            plt.xticks([])
            plt.savefig(
                f"{self.band.output}/BKGMAP_{self.data.body}_{self.band.name}.png"
            )
            plt.close()

            plt.imshow(sourcemask, origin="lower")
            plt.title(
                f"{self.band.output}/{self.data.body} {self.band.name} background map source masked"
            )
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.yticks([])
            plt.xticks([])
            plt.savefig(
                f"{self.band.output}/BKGMAP_SRCMASK_{self.data.body}_{self.band.name}.png"
            )
            plt.close()

        return self.data_hdu, self.err_hdu, None
