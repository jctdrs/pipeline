from typing import Optional

import numpy as np

import astropy
from astropy.wcs import WCS
from astropy.io import fits

from reproject import reproject_interp

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
        elif mode == "Automatic Differentiation":
            return RegridAutomaticDifferentiation(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> tuple[
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


class RegridSinglePass(Regrid):
    pass


class RegridMonteCarlo(Regrid):
    pass


class RegridAutomaticDifferentiation(Regrid):
    @staticmethod
    def compute_contributions_for_pixel(
        old_x,
        old_y,
        H_orig,
        W_orig,
    ):
        # Floor and ceil values to find the 4 nearest neighbors
        x0 = np.floor(old_x)
        x1 = x0.astype(int) + 1
        y0 = np.floor(old_y)
        y1 = y0.astype(int) + 1

        # Clip indices to ensure they are within bounds
        x0 = np.clip(x0, 0, W_orig - 1)
        x1 = np.clip(x1, 0, W_orig - 1)
        y0 = np.clip(y0, 0, H_orig - 1)
        y1 = np.clip(y1, 0, H_orig - 1)

        # Calculate distances for weights
        dx = old_x - x0
        dy = old_y - y0

        # Contributions based on bilinear interpolation
        w00 = (1 - dx) * (1 - dy)  # Top-left
        w10 = dx * (1 - dy)  # Top-right
        w01 = (1 - dx) * dy  # Bottom-left
        w11 = dx * dy  # Bottom-right

        # Collect indices and weights
        indices = np.array(
            [
                [old_y, old_x, y0, x0],
                [old_y, old_x, y0, x1],
                [old_y, old_x, y1, x0],
                [old_y, old_x, y1, x1],
            ]
        )
        contributions = np.array([w00, w10, w01, w11])

        return indices, contributions

    def run(
        self,
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        original_shape = self.err_hdu.data.shape
        with fits.open(self.task.parameters.target) as hdul:
            hdr_target = hdul[0].header
            target_shape = hdul[0].data.shape

        original_wcs = WCS(self.err_hdu.header)
        target_wcs = WCS(hdr_target)

        H_orig, W_orig = original_shape
        H_new, W_new = target_shape

        new_y, new_x = np.meshgrid(np.arange(H_new), np.arange(W_new), indexing="ij")

        # Convert the target pixel coordinates (new_y, new_x) to world coordinates
        target_coords = target_wcs.pixel_to_world(new_x, new_y)
        old_x, old_y = original_wcs.world_to_pixel(target_coords)

        indices = []
        contributions = []

        # Flatten the old_x and old_y arrays
        flat_old_x = old_x.flatten()
        flat_old_y = old_y.flatten()

        # Iterate over flattened arrays
        for x, y in zip(flat_old_x, flat_old_y):
            # Compute contributions for each pixel
            index, contribution = self.compute_contributions_for_pixel(
                x, y, H_orig, W_orig
            )
            indices.append(index)
            contributions.append(contribution)

        # Convert lists to NumPy arrays
        indices = np.array(indices)
        contributions = np.array(contributions)

        old_y = indices[:, :, 0]
        old_x = indices[:, :, 1]

        old_y = np.floor(old_y).astype(int)
        old_x = np.floor(old_x).astype(int)

        old_y = np.clip(old_y, 0, self.err_hdu.data.shape[0] - 1)
        old_x = np.clip(old_x, 0, self.err_hdu.data.shape[1] - 1)

        self.convert_from_Jyperpx_to_radiance(self.err_hdu)
        old_errors = self.err_hdu.data[old_y, old_x]

        sum_contributions = np.sum(contributions, axis=1, keepdims=True)
        contributions = contributions / sum_contributions

        weighted_contributions = np.square(old_errors) * np.square(contributions)

        self.err_hdu.data = np.sqrt(
            np.sum(weighted_contributions, axis=1).reshape(target_shape)
        )

        self.err_hdu.header.update(target_wcs.to_header())
        self.convert_from_radiance_to_Jyperpx(self.err_hdu)

        return self.data_hdu, self.err_hdu
