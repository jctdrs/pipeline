from setup import pipeline_validation
from setup import data_validation

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
    singleton_flux_list: list = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class Cutout(CutoutSingleton):
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
