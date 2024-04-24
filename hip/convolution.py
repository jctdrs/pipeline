import typing
from scipy.ndimage import zoom
import astropy
from astropy.convolution import convolve_fft
from util import read


class Convolution:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any],
        target: str,
        kernel: str,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.target = target
        self.kernel_path = kernel

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU, typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any]
    ]:
        # TODO: Check if target == input, then don't convolve
        self.setup_kernel()
        self.convolve()
        return self.data_hdu, self.err_hdu

    def setup_kernel(self) -> None:
        with astropy.io.fits.open(self.kernel_path) as hdul_kernel:
            self.kernel_hdu: astropy.io.fits.hdu.image.PrimaryHDU = hdul_kernel[0]
            self.rescale_kernel()
            self.convolve()
        return

    def rescale_kernel(self) -> None:
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
        return

    def convolve(self) -> None:
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
        return
