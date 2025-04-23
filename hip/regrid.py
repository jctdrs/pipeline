from typing import Optional
from typing import Tuple

import numpy as np

import astropy
from astropy.wcs import WCS
from astropy.io import fits

from reproject import reproject_interp

import matplotlib.pyplot as plt

from util import read


class Regrid:
    _instance = None
    _mode = None
    _band = None

    def __new__(cls, *args, **kwargs):
        mode = kwargs["task_control"].mode
        band = kwargs["band"].name
        if (
            cls._instance is None
            or (mode is None or mode != cls._mode)
            or (band is None or band != cls._band)
        ):
            cls._instance = super().__new__(cls)
            cls._mode = mode
            cls._band = band
        return cls._instance

    def __init__(
        self,
        task_control,
        data_hdu,
        err_hdu,
        data,
        task,
        band,
        instruments,
    ):
        self.task_control = task_control
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.data = data
        self.task = task
        self.band = band
        self.instruments = instruments

    @classmethod
    def create(cls, *args, **kwargs):
        mode = kwargs["task_control"].mode
        if mode == "Single Pass":
            return RegridSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return RegridMonteCarlo(*args, **kwargs)
        elif mode == "Analytic":
            return RegridAnalytic(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        self.convert_from_Jyperpx_to_radiance(self.data_hdu)
        with fits.open(self.task.parameters.target) as hdul:
            hdr_target = hdul[0].header

        wcs_out = WCS(hdr_target)
        self.data_hdu.data, _ = reproject_interp(
            input_data=self.data_hdu,
            output_projection=wcs_out,
        )
        self.data_hdu.header.update(wcs_out.to_header())
        self.convert_from_radiance_to_Jyperpx(self.data_hdu)
        return self.data_hdu, self.err_hdu

    def convert_from_Jyperpx_to_radiance(self, hdu) -> None:
        pixel_size = read.pixel_size_arcsec(hdu.header)
        px_x: float = pixel_size * 2 * np.pi / 360
        px_y: float = pixel_size * 2 * np.pi / 360

        # divide by 3.846x10^26 (Lsun in Watt) to convert W/Hz/m2/sr in
        # Lsun/Hz/m2/sr multiply by the galaxy distance in m2 to get Lsun/Hz/sr
        conversion_factor = (
            1e-26
            * pow((self.data.geometry.distance * 3.086e22), 2)
            * 4
            * np.pi
            / (px_x * px_y * 3.846e26)
        )

        hdu.data *= conversion_factor
        return None

    def convert_from_radiance_to_Jyperpx(self, hdu) -> None:
        pixel_size = read.pixel_size_arcsec(hdu.header)
        px_x: float = pixel_size * 2 * np.pi / 360
        px_y: float = pixel_size * 2 * np.pi / 360

        conversion_factor = (px_x * px_y * 3.846e26) / (
            1e-26 * pow((self.data.geometry.distance * 3.086e22), 2) * 4 * np.pi
        )

        hdu.data *= conversion_factor
        return None

    def diagnosis(self) -> None:
        if self.task.diagnosis:
            plt.imshow(self.data_hdu.data, origin="lower")
            plt.title(f"{self.data.body} {self.band.name} regrid map")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.yticks([])
            plt.xticks([])
            plt.savefig(
                f"{self.band.output}/REGRID_{self.data.body}_{self.band.name}.png"
            )
            plt.close()


class RegridSinglePass(Regrid):
    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        super().diagnosis()
        return self.data_hdu, self.err_hdu


class RegridMonteCarlo(Regrid):
    pass


class RegridAnalytic(Regrid):
    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        original_shape = self.err_hdu.data.shape
        with fits.open(self.task.parameters.target) as hdul:
            hdr_target = hdul[0].header

        original_wcs = WCS(self.err_hdu.header)
        target_wcs = WCS(hdr_target)

        # Create a meshgrid of target pixel indices (from (0, 0) to (W_target, H_target))
        H_target, W_target = target_wcs.pixel_shape
        target_x, target_y = np.meshgrid(np.arange(W_target), np.arange(H_target))

        # Convert target pixel coordinates to world coordinates
        target_coords = target_wcs.pixel_to_world(target_x, target_y)

        # Convert the adjusted target coordinates back to world coordinates in the input WCS
        input_coords = original_wcs.world_to_pixel(target_coords)

        # To propagate the error matrix properly, we need to consider the interpolation weights
        # Perform the same bilinear interpolation on the error matrix as done for the image
        # Calculate the weights based on bilinear interpolation for the target pixels
        x0 = np.floor(input_coords[0])
        x1 = x0 + 1
        y0 = np.floor(input_coords[1])
        y1 = y0 + 1

        # Clip indices to avoid out-of-bounds errors
        x0 = np.clip(x0, 0, original_shape[1] - 1)
        x1 = np.clip(x1, 0, original_shape[1] - 1)
        y0 = np.clip(y0, 0, original_shape[0] - 1)
        y1 = np.clip(y1, 0, original_shape[0] - 1)

        # Compute the weight for each pixel (based on distance to neighbors)
        dx = input_coords[0] - x0
        dy = input_coords[1] - y0
        w00 = (1 - dx) * (1 - dy)
        w10 = dx * (1 - dy)
        w01 = (1 - dx) * dy
        w11 = dx * dy

        self.convert_from_Jyperpx_to_radiance(self.err_hdu)
        # TODO: This is not valid propagation after convolution because of
        # the pixels are more correlated
        propagated_error_matrix = (
            np.square(w00)
            * np.square(self.err_hdu.data[y0.astype(int), x0.astype(int)])
            + np.square(w10)
            * np.square(self.err_hdu.data[y0.astype(int), x1.astype(int)])
            + np.square(w01)
            * np.square(self.err_hdu.data[y1.astype(int), x0.astype(int)])
            + np.square(w11)
            * np.square(self.err_hdu.data[y1.astype(int), x1.astype(int)])
        )

        self.err_hdu.data = np.sqrt(propagated_error_matrix)
        self.err_hdu.header.update(target_wcs.to_header())
        self.convert_from_radiance_to_Jyperpx(self.err_hdu)

        return self.data_hdu, self.err_hdu
