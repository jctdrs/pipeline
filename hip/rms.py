from typing import Optional

import copy

from util import read

import numpy as np
import numpy.ma as ma

import astropy
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy.stats import sigma_clipped_stats

from regions import PixCoord
from regions import EllipseAnnulusPixelRegion

import pyregion

import matplotlib.pyplot as plt


class RmsSingleton:
    _instance = None
    _mode = None

    def __new__(cls, *args, **kwargs):
        mode = kwargs["task_control"].mode
        if cls._instance is None and (mode is None or mode != cls._mode):
            cls._instance = super().__new__(cls)
            cls._mode = mode
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class Rms(RmsSingleton):
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
            return RmsSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return RmsMonteCarlo(*args, **kwargs)
        elif mode == "Automatic Differentiation":
            return RmsAutomaticDifferentiation(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        self.data_copy = copy.deepcopy(self.data_hdu.data)

        px_size = read.pixel_size_arcsec(self.data_hdu.header)
        wcs = WCS(self.data_hdu.header)
        ra_ = self.data.geometry.ra
        dec_ = self.data.geometry.dec
        rma_ = self.data.geometry.semiMajorAxis / 2
        rmi_ = rma_ / self.data.geometry.axialRatio
        self.rma = int(np.ceil(rma_ / px_size))
        self.rmi = int(np.ceil(rmi_ / px_size))
        self.position_px = wcs.all_world2pix(ra_, dec_, 0)

        region = """
                image
                ellipse({},{},{},{},{})
                """.format(
            self.position_px[0],
            self.position_px[1],
            self.rma,
            self.rmi,
            self.data.geometry.positionAngle,
        )
        r = pyregion.parse(region)
        mask_reg = r.get_mask(hdu=self.data_hdu)
        self.data_copy[mask_reg] = np.nan

        # define the ellipse annulus http://ds9.si.edu/doc/ref/region.html
        region_rms = """
                image
                ellipse({},{},{},{},{},{},{})
                """.format(
            self.position_px[0],
            self.position_px[1],
            1.25 * self.rma,
            1.25 * self.rmi,
            1.6 * self.rma,
            1.6 * self.rmi,
            self.data.geometry.positionAngle,
        )  # ref. Clarck et al. 2018, pag. 13, second paragraph

        r_rms = pyregion.parse(region_rms)
        mask_reg_rms = r_rms.get_mask(hdu=self.data_hdu)
        r_data = self.data_copy[mask_reg_rms]

        mean, median, self.std = sigma_clipped_stats(r_data, sigma=5)
        return self.data_hdu, self.err_hdu

    def diagnosis(self) -> None:
        if self.task.diagnosis:
            print(
                f"[INFO] RMS 5-sigma annulus {self.band.name} = {self.std:.03e} Jy/px"
            )
            # save the map in png where those regions are highlighted
            masked_data = ma.masked_invalid(self.data_copy)

            reg_ell_an = EllipseAnnulusPixelRegion(
                PixCoord(self.position_px[0], self.position_px[1]),
                inner_width=2.5 * self.rma,
                outer_width=3.2 * self.rma,
                inner_height=2.5 * self.rmi,
                outer_height=3.2 * self.rmi,
                angle=Angle(self.data.geometry.positionAngle, "deg"),
            )

            fig, ax = plt.subplots(1, 1)
            plt.imshow(masked_data, cmap="jet", origin="lower")
            plt.colorbar(label="Jy/px")
            reg_ell_an.plot(ax=ax, facecolor="none", edgecolor="yellow", lw=1)
            plt.xlabel("RA [px]")
            plt.ylabel("DEC [px]")
            plt.savefig(
                f"{self.band.output}ANNULUS_{self.data.body}_{self.band.name}.png"
            )
            plt.close()

        return None


class RmsAutomaticDifferentiation(Rms):
    pass


class RmsMonteCarlo(Rms):
    pass


class RmsSinglePass(Rms):
    def run(
        self,
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        super().diagnosis()
        return self.data_hdu, self.err_hdu
