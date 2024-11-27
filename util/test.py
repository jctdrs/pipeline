import typing

import jax.numpy as jnp

import astropy


class Test:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        name: str,
        body: str,
        geom: dict,
        instruments: dict,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.name = name
        self.body = body
        self.geom = geom
        self.instruments = instruments

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.array, typing.Any],
    ]:
        return self.data_hdu, self.err_hdu, None
