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

        a = [
            self.rma * 0.2,
            self.rma * 0.6,
            self.rma * 1.0,
            self.rma * 1.3,
            self.rma * 1.7,
            self.rma * 2.0,
            self.rma * 3.0,
            self.rma * 3.5,
            self.rma * 4.0,
            self.rma * 4.5,
            self.rma * 5.0,
            self.rma * 5.5,
            self.rma * 6.0,
            self.rma * 6.5,
            self.rma * 7.0,
            self.rma * 7.5,
            self.rma * 8.0,
            self.rma * 8.5,
            self.rma * 9.0,
            self.rma * 9.5,
            self.rma * 10.0,
        ]

        b = [
            self.rma * 0.2,
            self.rma * 0.6,
            self.rma * 1.0,
            self.rma * 1.3,
            self.rma * 1.7,
            self.rma * 2.0,
            self.rmi * 2.5,
            self.rmi * 3.0,
            self.rmi * 3.5,
            self.rmi * 4.0,
            self.rmi * 4.5,
            self.rmi * 5.0,
            self.rma * 5.5,
            self.rma * 6.0,
            self.rma * 6.5,
            self.rma * 7.0,
            self.rmi * 7.5,
            self.rmi * 8.0,
            self.rmi * 8.5,
            self.rmi * 9.0,
            self.rmi * 9.5,
            self.rmi * 10.0,
        ]

        apertures = [
            EllipticalAperture(
                self.position_px,
                a=ai,
                b=bi,
                theta=self.data.geometry.positionAngle,
            )
            for (ai, bi) in zip(a, b)
        ]
        phot_table = aperture_photometry(nan.data, apertures, error=None, mask=nan.mask)

        for col in phot_table.colnames:
            phot_table[col].info.format = "%.8g"

        integrated_flux_list = [phot_table[f"aperture_sum_{i}"] for i in range(21)]

        self.integrated_flux = np.interp(
            self.task.parameters.radius,
            np.arange(21),
            np.array(integrated_flux_list).reshape((-1,)),
        )

        return self.data_hdu, self.err_hdu

    def diagnosis(self) -> None:
        if self.task.diagnosis:
            # TODO: Use the integrated_L
            integrated_L = (  # noqa
                (self.integrated_flux * 1e-26)
                * pow((self.data.geometry.distance * 3.086e22), 2)
                * 4
                * np.pi
                / (3.846e26)
            )
            print(
                f"[INFO] Integrated flux {self.band.name} = {self.integrated_flux:.03f} Jy/px "
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
    pass


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
            cal_error = self.integrated_flux * self.band.calError / 100
            print(
                f"[INFO] Integrated flux error = {flux_error:.03f}_stat {cal_error:.03f}_inst {np.sqrt(flux_error**2 + cal_error**2):.03f}_tot"
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
