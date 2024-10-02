import math
import typing

import numpy as np
import numpy.ma as ma

import jax.numpy as jnp
from jax.scipy.signal import fftconvolve as jax_fftconvolve
from jax import jacfwd

from scipy.signal import fftconvolve
from scipy.ndimage import zoom

import astropy

from util import read


class Convolution:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        name: str,
        body: str,
        geom: dict,
        instruments: dict,
        use_jax: bool,
        kernel: str,
    ):
        self.data_hdu = data_hdu
        self.name = name
        self.body = body
        self.geom = geom
        self.instruments = instruments
        self.use_jax = use_jax
        self.kernel_path = kernel

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[np.ndarray, typing.Any],
    ]:
        mask = ma.masked_invalid(self.data_hdu.data).mask
        self.data_hdu.data[mask] = 0.0
        
        self.load_kernel()
        self.scale_kernel()
        self.normalize_kernel()
        self.convert_from_Jyperpx_to_radiance()

        if self.use_jax:
            grad_call = jacfwd(self.jax_convolve)
            grad_res = grad_call(jnp.array(self.data_hdu.data, dtype="bfloat16"))
            data = self.jax_convolve(jnp.array(self.data_hdu.data))
            self.data_hdu.data = np.array(data)
            self.convert_from_radiance_to_Jyperpx()
            del grad_call
            return self.data_hdu, grad_res

        else:
            self.convolve()
            self.convert_from_radiance_to_Jyperpx()
            self.data_hdu.data[mask] = np.nan

            return self.data_hdu, None

    def load_kernel(self) -> typing.Any:
        hdu_kernel: astropy.io.image.PrimaryHDU = astropy.io.fits.open(self.kernel_path)
        self.kernel_hdu = hdu_kernel[0]
        return None

    def scale_kernel(self) -> typing.Any:
        pixel_data = read.pixel_size_arcsec(self.data_hdu.header)
        pixel_kernel = read.pixel_size_arcsec(self.kernel_hdu.header)
        kernel_xsize, kernel_ysize = read.shape(self.kernel_hdu.header)

        # Resize kernel if necessary
        if pixel_kernel != pixel_data:
            ratio = pixel_kernel / pixel_data
            size = ratio * kernel_xsize
            # Ensure an odd kernel
            if round(size) % 2 == 0:
                size += 1
                ratio = size / kernel_xsize

            self.kernel_hdu.data = zoom(self.kernel_hdu.data, ratio) / ratio**2
        return None

    def normalize_kernel(self) -> typing.Any:
        self.kernel_hdu.data /= np.sum(self.kernel_hdu.data)
        return None

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

    def convolve(self) -> typing.Any:
        self.data_hdu.data = fftconvolve(
            self.data_hdu.data,
            self.kernel_hdu.data,
            mode="same",
        )
        return None

    def jax_convolve(self, im) -> jnp.array:
        data = jax_fftconvolve(im, self.kernel_hdu.data, mode="same")
        return data

    def convert_from_radiance_to_Jyperpx(self) -> typing.Any:
        pixel_size = read.pixel_size_arcsec(self.data_hdu.header)
        px_x: float = pixel_size * 2 * math.pi / 360
        px_y: float = pixel_size * 2 * math.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.geom["distance"] * 3.086e22), 2) * 4 * math.pi
        )

        self.data_hdu.data *= conversion_factor
        return None
