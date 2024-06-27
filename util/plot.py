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
        fig, ax = plt.subplots(2, 1, figsize=(7, 9))

        bound = np.argwhere(~np.isnan(self.data_hdu.data))
        ax[0].imshow(self.data_hdu.data, origin="lower")
        if bound.any():
            ax[0].set_xlim(min(bound[:, 1]), max(bound[:, 1]))
            ax[0].set_ylim(min(bound[:, 0]), max(bound[:, 0]))
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(self.err_hdu.data, origin="lower")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()

        return self.data_hdu, self.err_hdu, None
