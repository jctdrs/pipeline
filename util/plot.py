import typing

import jax.numpy as jnp

import astropy
import matplotlib.pyplot as plt


class PlotSingleton:
    _instance = None
    singleton_flux_list: list = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class Plot(PlotSingleton):
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

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.array, typing.Any],
    ]:
        plt.imshow(self.data_hdu.data, origin="lower")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()

        return self.data_hdu, self.err_hdu, None
