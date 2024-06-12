import typing

import astropy
import matplotlib.pyplot as plt


class Plot:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any],
        geom: dict,
        instruments: dict,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments

    def run(self):
        plt.imshow(self.data_hdu.data)
        plt.show()

        return self.data_hdu, self.err_hdu
