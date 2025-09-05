from typing import Optional
from typing import Tuple
from typing import List
from typing import Any

from astroquery.vizier import Vizier

from astropy.wcs import WCS
import astropy.units as au
import astropy

import numpy as np

from photutils.aperture import EllipticalAperture
from photutils.aperture import CircularAperture

from utilities import read

import matplotlib.pyplot as plt


class ForegroundMask:
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
            return ForegroundMaskSinglePass(*args, **kwargs)
        elif mode == "Monte-Carlo":
            return ForegroundMaskMonteCarlo(*args, **kwargs)
        elif mode == "Analytic":
            return ForegroundMaskAnalytic(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        px_size = read.pixel_size_arcsec(self.data_hdu.header)
        wcs = WCS(self.data_hdu.header)

        fgs_list = self.find_fgs()

        mask_gal_reg = self.get_mask_source()

        base_radius = [4.6, 3.0, 2.1, 1.4, 1.15, 0.7]
        resolution = self.instruments[self.band.name]["resolutionArcsec"]
        mask_factor = self.task.parameters.maskFactor
        r_masks = [(mask_factor * resolution * radius) / 2 for radius in base_radius]

        self.combined_mask = np.zeros(self.data_hdu.data.shape, dtype=bool)
        naxis1, naxis2 = self.data_hdu.header["NAXIS1"], self.data_hdu.header["NAXIS2"]

        # Create the mask for foreground sources
        for k, (fgs, r_mask) in enumerate(zip(fgs_list, r_masks)):
            if len(fgs) == 0:
                continue

            # Convert world coordinates to pixel coordinates
            fgs_pos_px = wcs.all_world2pix(fgs, 0)
            r_px = r_mask / px_size

            # Filter coordinates that are within bounds and not in galaxy region
            valid_coords = []
            for x_px, y_px in fgs_pos_px:
                x = int(np.ceil(x_px))
                y = int(np.ceil(y_px))
                if x < naxis1 and y < naxis2 and not mask_gal_reg[y, x]:
                    valid_coords.append((x_px, y_px))

            if not valid_coords:
                continue

            # Create aperture for all valid sources at once
            apertures = CircularAperture(valid_coords, r_px)
            masks = apertures.to_mask(method="exact")

            # Combine all masks for this magnitude bin
            for mask in masks:
                mask_img = mask.to_image(self.data_hdu.data.shape)
                if mask_img is not None:
                    self.combined_mask |= mask_img.astype(bool)

        # Apply the final combined mask
        self.data_hdu.data[self.combined_mask] = np.nan

        return self.data_hdu, self.err_hdu

    def find_fgs(self) -> List:
        magnitude_bins = ["<13.5", "<14.", "<15.5", "<16.", "<18.", "<40."]
        ra_trim = self.task.parameters.raTrim * au.arcmin
        dec_trim = self.task.parameters.decTrim * au.arcmin
        width = max(ra_trim * 2, dec_trim * 2)

        v = Vizier(
            columns=["RAJ2000", "DEJ2000", "Gmag"],
            catalog="I/355",
            row_limit=10000,
        )

        seen_sources = set()
        results = [np.empty((0, 2)) for _ in magnitude_bins]

        for idx, mag_filter in enumerate(magnitude_bins):
            try:
                fgs_sources = v.query_region(self.data.body, width=width)[0]
                coords = np.column_stack(
                    [fgs_sources["RAJ2000"], fgs_sources["DEJ2000"]]
                )
                new_sources = []
                for coord in coords:
                    coord_tuple = tuple(coord)
                    if coord_tuple not in seen_sources:
                        seen_sources.add(coord_tuple)
                        new_sources.append(coord)

                if new_sources:
                    results[idx] = np.array(new_sources)

            except Exception as e:
                print(f"[WARNING] Vizier query failed for magnitude {mag_filter}")
                continue

        return results

    def get_mask_source(self) -> Any:
        px_size = read.pixel_size_arcsec(self.data_hdu.header)

        wcs = WCS(self.data_hdu.header)

        position_px = wcs.all_world2pix(
            self.data.geometry.ra, self.data.geometry.dec, 0
        )

        rma = self.data.geometry.semiMajorAxis
        rmi = rma / self.data.geometry.axialRatio
        rma_px = rma / px_size
        rmi_px = rmi / px_size

        region = EllipticalAperture(
            position_px,
            a=rma_px,
            b=rmi_px,
            theta=np.deg2rad(self.data.geometry.positionAngle),
        )

        mask_reg = region.to_mask(method="exact").to_image(
            self.data_hdu.data.shape, dtype=bool
        )

        return mask_reg

    def diagnosis(self) -> None:
        if self.task.diagnosis:
            plt.imshow(self.combined_mask, origin="lower")
            plt.title(f"{self.data.body} {self.band.name} foregroundMask mask")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                f"{self.band.output}/FRG_MASK_{self.data.body}_{self.band.name}.png"
            )
            plt.close()

            plt.imshow(self.data_hdu.data, origin="lower")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel("Jy/px")
            plt.title(f"{self.data.body} {self.band.name} foregroundMask step result")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                f"{self.band.output}/FRG_DATA_{self.data.body}_{self.band.name}.png"
            )
            plt.close()

        return None


class ForegroundMaskSinglePass(ForegroundMask):
    def run(
        self,
    ) -> Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        super().run()
        super().diagnosis()
        return self.data_hdu, self.err_hdu


class ForegroundMaskMonteCarlo(ForegroundMask):
    pass


class ForegroundMaskAnalytic(ForegroundMask):
    pass
