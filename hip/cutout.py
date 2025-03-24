from typing import Optional

import matplotlib.pyplot as plt

import astropy
import astropy.units as au
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord, ICRS


class Cutout:
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
            return CutoutSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return CutoutMonteCarlo(*args, **kwargs)
        elif mode == "Analytic":
            return CutoutAnalytic(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        wcs_data = WCS(self.data_hdu.header)
        pos_center = SkyCoord(
            ra=self.data.geometry.ra * au.deg,
            dec=self.data.geometry.dec * au.deg,
            frame=ICRS,
        )
        sizeTrim = (
            self.task.parameters.decTrim * au.arcmin,
            self.task.parameters.raTrim * au.arcmin,
        )

        data_cutout = Cutout2D(
            self.data_hdu.data, position=pos_center, size=sizeTrim, wcs=wcs_data
        )
        self.data_hdu.data = data_cutout.data
        self.data_hdu.header.update(data_cutout.wcs.to_header())

        if self.err_hdu is not None:
            wcs_err = WCS(self.err_hdu.header)
            err_cutout = Cutout2D(
                self.err_hdu.data, position=pos_center, size=sizeTrim, wcs=wcs_err
            )
            self.err_hdu.data = err_cutout.data
            self.err_hdu.header.update(err_cutout.wcs.to_header())

        return self.data_hdu, self.err_hdu

    def diagnosis(self) -> None:
        if self.task.diagnosis:
            plt.imshow(self.data_hdu.data, origin="lower")
            plt.title(f"{self.data.body} {self.band.name} cutout")
            plt.xticks([])
            plt.yticks([])
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.savefig(
                f"{self.band.output}/CUTOUT_{self.data.body}_{self.band.name}.png"
            )
            plt.close()
        return None


class CutoutMonteCarlo(Cutout):
    pass


class CutoutAnalytic(Cutout):
    pass


class CutoutSinglePass(Cutout):
    def run(
        self,
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        super().diagnosis()
        return self.data_hdu, self.err_hdu
