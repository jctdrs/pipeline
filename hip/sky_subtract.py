from typing import Optional
from typing import Tuple

import astropy
from astropy.stats import SigmaClip
from astropy.wcs import WCS

from photutils import background

import numpy as np

import pyregion

import copy

import numpy.ma as ma
import matplotlib.pyplot as plt

from util import read


class SkySubtract:
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
            return SkySubtractSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return SkySubtractMonteCarlo(*args, **kwargs)
        elif mode == "Analytic":
            return SkySubtractAnalytic(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        # This masking is needed to tame the Warning from photutils
        self.data_hdu_invalid = ma.masked_invalid(self.data_hdu.data)
        self.data_hdu.data[self.data_hdu_invalid.mask] = 0.0

        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        xsize, ysize = read.shape(self.data_hdu.header)
        lcell_px = np.ceil(
            self.task.parameters.cellFactor
            * self.instruments[self.band.name]["resolutionArcsec"]
            / pixel_size
        )
        ncells1 = int(np.ceil(xsize / lcell_px))
        ncells2 = int(np.ceil(ysize / lcell_px))

        wcs = WCS(self.data_hdu.header)
        pos = wcs.all_world2pix(self.data.geometry.ra, self.data.geometry.dec, 0)
        rma = np.ceil(self.data.geometry.semiMajorAxis / 2 / pixel_size)
        rmi = np.ceil(
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

        self.bkg_mask = region.get_mask(hdu=self.data_hdu)

        self.bkg = background.Background2D(
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
            mask=self.bkg_mask,
            exclude_percentile=10.0,
            bkg_estimator=background.SExtractorBackground(),
            bkgrms_estimator=background.StdBackgroundRMS(),
        )
        self.data_hdu.data = self.data_hdu.data - self.bkg.background
        self.data_hdu.data[self.data_hdu_invalid.mask] = np.nan

        return self.data_hdu, self.err_hdu

    def diagnosis(self) -> None:
        if self.task.diagnosis:
            mask_bkg = copy.copy(self.bkg.background)
            mask_bkg[self.data_hdu_invalid.mask] = np.nan

            sourcemask = copy.deepcopy(mask_bkg)
            sourcemask[self.bkg_mask] = np.nan

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

            plt.imshow(self.bkg.background_rms, origin="lower")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.yticks([])
            plt.xticks([])
            plt.savefig(
                f"{self.band.output}/BKGMAP_RMS_{self.data.body}_{self.band.name}.png"
            )
            plt.close()

            plt.imshow(sourcemask, origin="lower")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.yticks([])
            plt.xticks([])
            self.bkg.plot_meshes(outlines=True, marker=".", color="red", alpha=0.3)
            plt.savefig(
                f"{self.band.output}/BKGMAP_MESHES_{self.data.body}_{self.band.name}.png"
            )
            plt.close()

        return None


class SkySubtractMonteCarlo(SkySubtract):
    pass


class SkySubtractAnalytic(SkySubtract):
    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        self.err_hdu.data = np.sqrt(
            np.square(self.err_hdu.data) + np.square(self.bkg.background_rms)
        )
        return self.data_hdu, self.err_hdu


class SkySubtractSinglePass(SkySubtract):
    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        super().diagnosis()
        return self.data_hdu, self.err_hdu
