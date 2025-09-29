from typing import Optional
from typing import Tuple

from utilities import read

import numpy as np
import numpy.ma as ma

import astropy
from astropy.wcs import WCS

from photutils.aperture import EllipticalAperture
from photutils.aperture import aperture_photometry

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
    ):
        self.task_control = task_control
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.data = data
        self.task = task
        self.band = band

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
        ra_ = self.data.geometry.ra
        dec_ = self.data.geometry.dec
        rma = self.data.geometry.semiMajorAxis
        rmi = rma / self.data.geometry.axialRatio
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        self.position_px = wcs.all_world2pix(ra_, dec_, 0)
        self.rma_px = rma / pixel_size
        self.rmi_px = rmi / pixel_size
        nan = ma.masked_invalid(self.data_hdu.data)

        self.aperture = EllipticalAperture(
            self.position_px,
            a=self.task.parameters.sizeFactor * self.rma_px,
            b=self.task.parameters.sizeFactor * self.rmi_px,
            theta=np.deg2rad(self.data.geometry.positionAngle),
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
                f"[INFO] Instrumental integrated flux error {self.band.name} = {cal_error:.03f} Jy/px "
            )

            fig, ax = plt.subplots(1, 1)
            bound = np.argwhere(~np.isnan(self.data_hdu.data))
            if bound.any():
                ax.set_xlim(float(np.min(bound[:, 1])), float(np.max(bound[:, 1])))
                ax.set_ylim(float(np.min(bound[:, 0])), float(np.max(bound[:, 0])))

            plt.imshow(self.data_hdu.data, origin="lower")
            plt.title(f"{self.data.body} {self.band.name} integrate step result")
            self.aperture.plot(
                ax=ax,
                facecolor="none",
                edgecolor="red",
                lw=1,
                label="Aperture",
            )
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.xticks([])
            plt.yticks([])
            plt.legend(loc="lower left")
            plt.savefig(
                f"{self.band.output}/INTEGRATE_{self.data.body}_{self.band.name}.png"
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
        ra_ = self.data.geometry.ra
        dec_ = self.data.geometry.dec
        rma = self.data.geometry.semiMajorAxis
        rmi = rma / self.data.geometry.axialRatio
        self.position_px = wcs.all_world2pix(ra_, dec_, 0)
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)

        self.rma_px = rma / pixel_size
        self.rmi_px = rmi / pixel_size

        self.aperture = EllipticalAperture(
            self.position_px,
            a=self.task.parameters.sizeFactor * self.rma_px,
            b=self.task.parameters.sizeFactor * self.rmi_px,
            theta=np.deg2rad(self.data.geometry.positionAngle),
        )

        phot_table = aperture_photometry(
            data=self.err_hdu.data,
            error=self.err_hdu.data,
            apertures=self.aperture,
        )

        if self.err_hdu.header["ERRCORR"] == "False":
            self.integrated_flux_error = phot_table["aperture_sum_err"][0]
        else:
            self.integrated_flux_error = phot_table["aperture_sum"][0]

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
