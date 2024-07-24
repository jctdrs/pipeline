import typing

import numpy as np

import astropy
import matplotlib.pyplot as plt


class Plot:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        geom: dict,
        instruments: dict,
        use_jax: bool,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.use_jax = use_jax

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[np.ndarray, typing.Any],
    ]:

        bound = np.argwhere(~np.isnan(self.data_hdu.data))
        plt.imshow(self.data_hdu.data, origin="lower")
        if bound.any():
            plt.xlim(min(bound[:, 1]), max(bound[:, 1]))
            plt.ylim(min(bound[:, 0]), max(bound[:, 0]))
        plt.xticks([])
        plt.yticks([])
        plt.show()

        return self.data_hdu, self.err_hdu, None
