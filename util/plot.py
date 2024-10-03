import typing

import numpy as np

import astropy
import matplotlib.pyplot as plt


class Plot:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        name: str,
        body: str,
        geom: dict,
        instruments: dict,
        use_jax: bool,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.name = name
        self.body = body
        self.geom = geom
        self.instruments = instruments
        self.use_jax = use_jax

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[np.ndarray, typing.Any],
    ]:
        plt.imshow(self.data_hdu.data, origin="lower")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()

        return self.data_hdu, self.err_hdu, None
