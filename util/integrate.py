import math
import typing

import numpy as np
import numpy.ma as ma

import astropy
from astropy.wcs import WCS
from astropy.coordinates import Angle  # noqa

from photutils.aperture.ellipse import EllipticalAperture
from photutils.aperture import aperture_photometry

from regions import PixCoord, EllipsePixelRegion  # noqa

import matplotlib.pyplot as plt  # noqa

from util import read


class IntegrateSingleton:
    _instance = None
    flux_error: float = 0
    mean: float = 0
    count: float = 1
    M2: float = 0

    check = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def update(cls, value):
        cls.check.append(value)
        delta = value - cls.mean
        cls.mean += delta / cls.count
        delta2 = value - cls.mean
        cls.M2 += delta * delta2
        cls.flux_error = np.sqrt(cls.M2 / cls.count)
        cls.count += 1

    def run(self, *args, **kwargs):
        pass


class Integrate(IntegrateSingleton):
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        name: str,
        body: str,
        geom: dict,
        instruments: dict,
        diagnosis: bool,
        MC_diagnosis: bool,
        differentiate: bool,
        radius: float,
        calError: float,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.name = name
        self.body = body
        self.geom = geom
        self.instruments = instruments
        self.diagnosis = diagnosis
        self.MC_diagnosis = MC_diagnosis
        self.differentiate = differentiate
        self.radius = radius
        self.cal_error = calError

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[np.ndarray, typing.Any],
    ]:
        wcs = WCS(self.data_hdu.header)
        px_size = read.pixel_size_arcsec(self.data_hdu.header)
        ra_ = self.geom["ra"]
        dec_ = self.geom["dec"]
        rma_ = self.geom["semiMajorAxis"] / 2
        rmi_ = rma_ / self.geom["axialRatio"]
        position_px = wcs.all_world2pix(ra_, dec_, 0)
        rma = int(math.ceil(rma_ / px_size))
        rmi = int(math.ceil(rmi_ / px_size))
        nan = ma.masked_invalid(self.data_hdu.data)

        a = [
            int(rma * 0.2),
            int(rma * 0.6),
            int(rma * 1.0),
            int(rma * 1.3),
            int(rma * 1.7),
            int(rma * 2.0),
            int(rma * 3.0),
            int(rma * 3.5),
            int(rma * 4.0),
            int(rma * 4.5),
            int(rma * 5.0),
            int(rma * 5.5),
            int(rma * 6.0),
            int(rma * 6.5),
            int(rma * 7.0),
            int(rma * 7.5),
            int(rma * 8.0),
            int(rma * 8.5),
            int(rma * 9.0),
            int(rma * 9.5),
            int(rma * 10.0),
        ]

        b = [
            int(rma * 0.2),
            int(rma * 0.6),
            int(rma * 1.0),
            int(rma * 1.3),
            int(rma * 1.7),
            int(rma * 2.0),
            int(rmi * 2.5),
            int(rmi * 3.0),
            int(rmi * 3.5),
            int(rmi * 4.0),
            int(rmi * 4.5),
            int(rmi * 5.0),
            int(rma * 5.5),
            int(rma * 6.0),
            int(rma * 6.5),
            int(rma * 7.0),
            int(rmi * 7.5),
            int(rmi * 8.0),
            int(rmi * 8.5),
            int(rmi * 9.0),
            int(rmi * 9.5),
            int(rmi * 10.0),
        ]

        apertures = [
            EllipticalAperture(
                position_px,
                a=ai,
                b=bi,
                theta=self.geom["positionAngle"],
            )
            for (ai, bi) in zip(a, b)
        ]
        phot_table = aperture_photometry(nan.data, apertures, error=None, mask=nan.mask)

        for col in phot_table.colnames:
            phot_table[col].info.format = "%.8g"

        integrated_flux_list = [phot_table[f"aperture_sum_{i}"] for i in range(21)]

        integrated_flux = np.interp(
            self.radius,
            np.arange(21),
            np.array(integrated_flux_list).reshape((-1,)),
        )

        self.update(integrated_flux)

        if self.diagnosis:
            integrated_L = (
                (integrated_flux * 1e-26)
                * pow((self.geom["distance"] * 3.086e22), 2)
                * 4
                * math.pi
                / (3.846e26)
            )
            print(f"[INFO]  Integrated flux = {integrated_flux:.03f}")

            reg = EllipsePixelRegion(
                PixCoord(position_px[0], position_px[1]),
                width=self.radius * rma,
                height=self.radius * rmi,
                angle=Angle(self.geom["positionAngle"], "deg"),
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
                label=f"radius={self.radius}",
            )
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.xticks([])
            plt.yticks([])
            plt.legend()
            plt.savefig(f"PHOT_{self.body}_{self.name}.png")
            plt.close()

        if self.MC_diagnosis:
            cal_error = integrated_flux * self.cal_error / 100
            print(
                f"[INFO]  Integrated flux error = {self.flux_error:.03f}_stat {cal_error:.03f}_inst {np.sqrt(self.flux_error**2 + cal_error**2):0.3f}_tot"
            )

        return self.data_hdu, self.err_hdu, None
