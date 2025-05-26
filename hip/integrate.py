from typing import Optional
from typing import Tuple

from util import read

import numpy as np
import numpy.ma as ma

import astropy
from astropy.wcs import WCS
from astropy.coordinates import Angle

from photutils.aperture.ellipse import EllipticalAperture
from photutils.aperture import aperture_photometry

from regions import PixCoord, EllipsePixelRegion

import matplotlib.pyplot as plt


class Integrate:
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
            return IntegrateSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return IntegrateMonteCarlo(*args, **kwargs)
        elif mode == "Analytic":
            return IntegrateAnalytic(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        wcs = WCS(self.data_hdu.header)
        px_size = read.pixel_size_arcsec(self.data_hdu.header)
        ra_ = self.data.geometry.ra
        dec_ = self.data.geometry.dec
        rma_ = self.data.geometry.semiMajorAxis / 2
        rmi_ = rma_ / self.data.geometry.axialRatio
        self.position_px = wcs.all_world2pix(ra_, dec_, 0)
        self.rma = int(np.ceil(rma_ / px_size))
        self.rmi = int(np.ceil(rmi_ / px_size))
        nan = ma.masked_invalid(self.data_hdu.data)

        self.aperture = EllipticalAperture(
            self.position_px,
            a=self.task.parameters.radius * self.rma,
            b=self.task.parameters.radius * self.rmi,
            theta=self.data.geometry.positionAngle,
        )

        phot_table = aperture_photometry(
            nan.data, self.aperture, error=None, mask=nan.mask
        )

        self.integrated_flux = phot_table["aperture_sum"].value[0]

        return self.data_hdu, self.err_hdu

    def diagnosis(self) -> None:
        if self.task.diagnosis:
            print(
                f"[INFO] Integrated flux {self.band.name} = {self.integrated_flux:.03f} Jy/px "
            )
            cal_error = self.integrated_flux * self.band.calError / 100

            print(
                f"[INFP] Instrumental integrated flux error {self.band.name} = {cal_error:.03f} Jy/px "
            )

            reg = EllipsePixelRegion(
                PixCoord(self.position_px[0], self.position_px[1]),
                width=self.task.parameters.radius * self.rma,
                height=self.task.parameters.radius * self.rmi,
                angle=Angle(self.data.geometry.positionAngle, "deg"),
            )

            fig, ax = plt.subplots(1, 1)
            bound = np.argwhere(~np.isnan(self.data_hdu.data))
            if bound.any():
                ax.set_xlim(min(bound[:, 1]), max(bound[:, 1]))
                ax.set_ylim(min(bound[:, 0]), max(bound[:, 0]))

            plt.imshow(self.data_hdu.data, origin="lower")
            reg.plot(
                ax=ax,
                facecolor="none",
                edgecolor="red",
                lw=1,
                label=f"radius={self.task.parameters.radius}",
            )
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.xticks([])
            plt.yticks([])
            plt.legend()
            plt.savefig(
                f"{self.band.output}/PHOT_{self.data.body}_{self.band.name}.png"
            )
            plt.close()
        return None


class IntegrateAnalytic(Integrate):
    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        wcs = WCS(self.err_hdu.header)
        px_size = read.pixel_size_arcsec(self.err_hdu.header)
        ra_ = self.data.geometry.ra
        dec_ = self.data.geometry.dec
        rma_ = self.data.geometry.semiMajorAxis / 2
        rmi_ = rma_ / self.data.geometry.axialRatio
        self.position_px = wcs.all_world2pix(ra_, dec_, 0)
        self.rma = int(np.ceil(rma_ / px_size))
        self.rmi = int(np.ceil(rmi_ / px_size))

        self.aperture = EllipticalAperture(
            self.position_px,
            a=self.task.parameters.radius * self.rma,
            b=self.task.parameters.radius * self.rmi,
            theta=self.data.geometry.positionAngle,
        )

        phot_table = aperture_photometry(
            data=self.err_hdu.data,
            error=self.err_hdu.data,
            apertures=self.aperture,
        )

        self.integrated_flux_error = phot_table["aperture_sum_err"][0]

        print(
            f"[INFO] Statistical integrated flux error {self.band.name} = "
            f"{self.integrated_flux_error:.03f} Jy/px"
        )
        return self.data_hdu, self.err_hdu


class IntegrateMonteCarlo(Integrate):
    count = 0
    mean = 0.0
    M2 = 0.0

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        self.update(self.integrated_flux)

        idx = self.task_control.idx
        if self.task_control.MC_diagnosis[idx]:
            flux_error = np.sqrt(self.M2 / (self.count - 1))
            print(
                f"[INFO] Statistical integrated flux error {self.band.name} = {flux_error:.03f} Jy/px"
            )
        return self.data_hdu, self.err_hdu


class IntegrateSinglePass(Integrate):
    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        super().diagnosis()
        return self.data_hdu, self.err_hdu
