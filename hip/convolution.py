import math
import typing

from util import read

from scipy.ndimage import zoom
from astropy.convolution import convolve_fft
import astropy


class Convolution:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any],
        geom: dict,
        instruments: dict,
        name: str,
        kernel: str,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments
        self.name = name
        self.kernel_path = kernel

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU, typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any]
    ]:
        self.setup_kernel()
        self.convert_from_Jyperpx_to_radiance()
        self.convolve()
        self.convert_from_radiance_to_Jyperpx()
        return self.data_hdu, self.err_hdu

    def setup_kernel(self) -> typing.Any:
        with astropy.io.fits.open(self.kernel_path) as hdul_kernel:
            self.kernel_hdu: astropy.io.fits.hdu.image.PrimaryHDU = hdul_kernel[0]
            self.scale_kernel()
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

    def convert_from_Jyperpx_to_radiance(self) -> typing.Any:
        pixel_size = read.pixel_size(self.data_hdu.header)
        px_x: float = pixel_size[0] * 2 * math.pi / 360
        px_y: float = pixel_size[1] * 2 * math.pi / 360

        # divide by 3.846x10^26 (Lsun in Watt) to convert W/Hz/m2/sr in
        # Lsun/Hz/m2/sr multiply by the galaxy distance in m2 to get Lsun/Hz/sr
        conversion_factor = 1e-26 * pow((self.geom["distance"] * 3.086e22), 2) * 4 * math.pi / (px_x * px_y * 3.846e26)

        self.data_hdu.data *= conversion_factor
        return None

    def convolve(self) -> typing.Any:
        self.data_hdu.data = convolve_fft(
            self.data_hdu.data,
            self.kernel_hdu.data,
            nan_treatment="interpolate",
            normalize_kernel=True,
            preserve_nan=True,
            fft_pad=True,
            boundary="fill",
            fill_value=0.0,
            allow_huge=True,
        )
        return None

    def convert_from_radiance_to_Jyperpx(self) -> typing.Any:
        pixel_size = read.pixel_size(self.data_hdu.header)
        px_x: float = pixel_size[0] * 2 * math.pi / 360
        px_y: float = pixel_size[1] * 2 * math.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.geom["distance"] * 3.086e22), 2) * 4 * math.pi
        )

        self.data_hdu.data *= conversion_factor
        return None
