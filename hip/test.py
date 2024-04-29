import typing
import time

import astropy


class Test:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any],
        geom: dict,
        instruments: dict,
        test: str,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.test = test

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU, typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any]
    ]:
        time.sleep(1)
        return self.data_hdu, self.err_hdu
