import math
import typing

import numpy as np

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
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        geom: dict,
        instruments: dict,
        use_jax: bool,
        name: str,
        kernel: str,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.use_jax = use_jax
        self.name = name
        self.kernel_path = kernel

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[np.ndarray, typing.Any],
    ]:
        self.setup_kernel()
        self.convert_from_Jyperpx_to_radiance()

        if self.use_jax:
            grad_call = jacfwd(self.jax_convolve)
            grad_res = grad_call(jnp.array(self.data_hdu.data, dtype='bfloat16'))
            data = self.jax_convolve(jnp.array(self.data_hdu.data), dtype='bfloat16')
            self.data_hdu.data = np.array(data)
            self.convert_from_radiance_to_Jyperpx()
            del grad_call
            return self.data_hdu, self.err_hdu, grad_res

        else:
            self.convolve()
            self.convert_from_radiance_to_Jyperpx()
            return self.data_hdu, self.err_hdu, None

    def setup_kernel(self) -> typing.Any:
        with astropy.io.fits.open(self.kernel_path) as hdul_kernel:
            self.kernel_hdu: astropy.io.fits.hdu.image.PrimaryHDU = hdul_kernel[
                0
            ]
            self.crop_kernel()
            self.scale_kernel()
            self.normalize_kernel()
        return None

    def scale_kernel(self) -> typing.Any:
        pixel_data = read.pixel_scale(self.data_hdu.header)
        pixel_kernel = read.pixel_scale(self.kernel_hdu.header)
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

    def crop_kernel(self) -> typing.Any:
        xsize_d = self.data_hdu.data.shape[0]
        ysize_d = self.data_hdu.data.shape[1]

        xsize_k = self.kernel_hdu.data.shape[0]
        ysize_k = self.kernel_hdu.data.shape[1]

        self.kernel_hdu.data = self.kernel_hdu.data[
            xsize_k // 2 - xsize_d // 2 : xsize_k // 2 + xsize_d // 2,
            ysize_k // 2 - ysize_d // 2 : ysize_k // 2 + ysize_d // 2,
        ]
        return None

    def normalize_kernel(self) -> typing.Any:
        self.kernel_hdu.data /= np.sum(self.kernel_hdu.data)
        return None

    def convert_from_Jyperpx_to_radiance(self) -> typing.Any:
        pixel_size = read.pixel_size(self.data_hdu.header)
        px_x: float = pixel_size[0] * 2 * math.pi / 360
        px_y: float = pixel_size[1] * 2 * math.pi / 360

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
        pixel_size = read.pixel_size(self.data_hdu.header)
        px_x: float = pixel_size[0] * 2 * math.pi / 360
        px_y: float = pixel_size[1] * 2 * math.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.geom["distance"] * 3.086e22), 2) * 4 * math.pi
        )

        self.data_hdu.data *= conversion_factor
        return None
