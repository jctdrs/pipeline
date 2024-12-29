import typing

import jax.numpy as jnp

import matplotlib.pyplot as plt

import astropy
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as au
from astropy.wcs import WCS


class CutoutSingleton:
    _instance = None
    _mode = None

    def __new__(cls, *args, **kwargs):
        mode = kwargs["mode"]
        if cls._instance is None and (mode is None or mode != cls._mode):
            cls._instance = super().__new__(cls)
            cls._mode = mode
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class Cutout(CutoutSingleton):
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def create(cls, *args, **kwargs):
        mode = kwargs["mode"]
        if mode == "Single Pass":
            return CutoutSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return CutoutMonteCarlo(*args, **kwargs)
        elif mode == "Automatic Differentiation":
            return CutoutAutomaticDifferentiation(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.array, typing.Any],
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

        return self.data_hdu, self.err_hdu, None


class CutoutMonteCarlo(Cutout):
    pass


class CutoutAutomaticDifferentiation(Cutout):
    pass


class CutoutSinglePass(Cutout):
    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.ndarray, typing.Any],
    ]:
        super().run()

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

        return self.data_hdu, self.err_hdu, None
