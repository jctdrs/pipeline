import math
import typing

import numpy as np

import jax.numpy as jnp
from jax import jacfwd
from jax.scipy.ndimage import map_coordinates

import astropy
from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs.utils import pixel_to_pixel

from reproject import reproject_interp

from util import read

class ReprojectSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class Reproject(ReprojectSingleton):
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
        target: str,
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
        self.target = target

    def pixel_to_pixel_with_roundtrip(self, wcs1, wcs2, *inputs):
        inputs = np.array(inputs)
        outputs = pixel_to_pixel(wcs1, wcs2, *inputs)

        # Now convert back to check that coordinates round-trip, if not then set to NaN
        inputs_check = pixel_to_pixel(wcs2, wcs1, *outputs)
        reset = jnp.zeros(inputs_check[0].shape, dtype=bool)
        for ipix in range(len(inputs_check)):
            reset |= jnp.abs(inputs_check[ipix] - inputs[ipix]) > 1
        if jnp.any(reset):
            for ipix in range(len(inputs_check)):
                outputs[ipix] = outputs[ipix].copy()
                outputs[ipix][reset] = jnp.nan

        return outputs

    def setup_interpolate(self):
        wcs_in = WCS(self.data_hdu.header)

        with fits.open(self.target) as hdul:
            hdr_target = hdul[0].header

        wcs_out = WCS(hdr_target)
        shape_out = wcs_out.low_level_wcs.array_shape

        wcs_dims = shape_out[-wcs_in.low_level_wcs.pixel_n_dim :]
        pixel_out = jnp.meshgrid(
            *[jnp.arange(size, dtype=float) for size in wcs_dims],
            indexing="ij",
            sparse=False,
            copy=True,
        )

        pixel_out = [p.ravel() for p in pixel_out]
        pixel_in = self.pixel_to_pixel_with_roundtrip(
            wcs_out, wcs_in, *pixel_out[::-1]
        )[::-1]
        pixel_in = jnp.array(pixel_in)

        # Interpolate array on to the pixels coordinates in pixel_in
        pixel_in = jnp.array(pixel_in)

        grad_call = jacfwd(self.interpolate, argnums=0)
        grad_res = grad_call(jnp.array(self.data_hdu.data, dtype="bfloat16"), pixel_in)
        data = self.interpolate(
            jnp.array(self.data_hdu.data, dtype="bfloat16"), pixel_in
        )
        self.data_hdu.data = jnp.array(data, dtype="float32")
        self.data_hdu.header.update(wcs_out.to_header())

        return grad_res

    def interpolate(self, array_in, pixel_in):
        array_out = map_coordinates(
            array_in, pixel_in, order=1, cval=jnp.nan, mode="constant"
        )
        array_out = jnp.reshape(
            array_out,
            (int(array_out.shape[0] ** 0.5), int(array_out.shape[0] ** 0.5)),
        )
        return array_out

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.array, typing.Any],
    ]:
        self.convert_from_Jyperpx_to_radiance()

        if self.differentiate:
            grad_res = self.setup_interpolate()
            self.convert_from_radiance_to_Jyperpx()
            # TODO: need crop ?
            # self.crop()
            return self.data_hdu, self.err_hdu, grad_res
        else:
            self.reproject()
            self.convert_from_radiance_to_Jyperpx()
            # self.crop()
            return self.data_hdu, self.err_hdu, None

    def convert_from_Jyperpx_to_radiance(self) -> typing.Any:
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        px_x: float = pixel_size * 2 * math.pi / 360
        px_y: float = pixel_size * 2 * math.pi / 360

        # divide by 3.846x10^26 (Lsun in Watt) to convert W/Hz/m2/sr in
        # Lsun/Hz/m2/sr multiply by the galaxy distance in m2 to get Lsun/Hz/sr
        conversion_factor = (
            1e-26
            * pow((self.geom["distance"] * 3.086e22), 2)
            * 4
            * math.pi
            / (px_x * px_y * 3.846e26)
        )

        self.data_hdu.data *= conversion_factor
        return None

    def reproject(self) -> typing.Any:
        with fits.open(self.target) as hdul:
            hdr_target = hdul[0].header

        wcs = WCS(hdr_target)
        self.data_hdu.data, _ = reproject_interp(
            input_data=self.data_hdu,
            output_projection=wcs,
        )
        self.data_hdu.header.update(wcs.to_header())

        return None

    def convert_from_radiance_to_Jyperpx(self) -> typing.Any:
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        px_x: float = pixel_size * 2 * math.pi / 360
        px_y: float = pixel_size * 2 * math.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.geom["distance"] * 3.086e22), 2) * 4 * math.pi
        )

        self.data_hdu.data *= conversion_factor
        return None

    def crop(self) -> typing.Any:
        bound = jnp.argwhere(~jnp.isnan(self.data_hdu.data))
        if bound.any():
            self.data_hdu.data = self.data_hdu.data[
                min(bound[:, 0]) : max(bound[:, 0]),
                min(bound[:, 1]) : max(bound[:, 1]),
            ]
        return None
