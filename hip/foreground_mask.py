from typing import Optional

from astroquery.vizier import Vizier

from astropy.wcs import WCS
import astropy.units as au
import astropy

import numpy as jnp

import pyregion


from util import read


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
        elif mode == "Automatic Differentiation":
            return ForegroundMaskAutomaticDifferentiation(*args, **kwargs)
        else:
            msg = f"[ERROR] Mode '{mode}' not recognized."
            raise ValueError(msg)

    def run(
        self,
    ) -> tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        Optional[astropy.io.fits.hdu.image.PrimaryHDU],
    ]:
        # Generate copy to be masked
        px_size = read.pixel_size_arcsec(self.data_hdu.header)
        wcs = WCS(self.data_hdu.header)

        fgs_list, rad_fac = self.find_fgs()

        # Exclude the central regions of the galaxy from the subtraction
        mask_gal_reg = self.get_mask_source()

        # Precompute the radius scaling factors for all FGS sources
        r_masks = [
            (
                rad_fac[k]
                * self.task.parameters.factor
                * self.instruments[self.band.name]["resolutionArcsec"]
            )
            // 2
            for k in range(len(fgs_list))
        ]

        # Create the mask for foreground sources
        for k, fgs in enumerate(fgs_list):
            # Convert world coordinates to pixel coordinates
            fgs_pos_px = wcs.all_world2pix(fgs, 0)
            r_mask = r_masks[k]

            # Iterate over the points and apply the mask
            for i in range(len(fgs_pos_px)):
                x = jnp.ceil(fgs_pos_px[i][0]).astype(int)
                y = jnp.ceil(fgs_pos_px[i][1]).astype(int)

                # Check if the pixel is within bounds and not part of the galaxy region
                if (
                    x < self.data_hdu.header["NAXIS1"]
                    and y < self.data_hdu.header["NAXIS2"]
                    and not mask_gal_reg[y, x]
                ):
                    # Create the circular mask region
                    r_circle = int(jnp.ceil(r_mask / px_size))
                    region = f"""
                        image
                        circle({fgs_pos_px[i][0]},{fgs_pos_px[i][1]},{r_circle})
                    """
                    r = pyregion.parse(region)
                    mask_fgs_reg = r.get_mask(hdu=self.data_hdu)

                    # Apply the mask by setting the corresponding pixels to NaN
                    self.data_hdu.data[mask_fgs_reg] = jnp.nan

        return self.data_hdu, self.err_hdu

    def find_fgs(self) -> tuple[list, list]:
        mag = ["<13.5", "<14.", "<15.5", "<16.", "<18.", "<40."]
        rad_fac = [4.6, 3.0, 2.1, 1.4, 1.15, 0.7]

        sizeTrim = (
            self.task.parameters.decTrim * au.arcmin,
            self.task.parameters.raTrim * au.arcmin,
        )

        fgs_set = set()
        fgs_list = []

        for k in range(len(mag)):
            v = Vizier(
                columns=["RAJ2000", "DEJ2000"],
                catalog="I/355",
                row_limit=10000,
                column_filters={"Gmag": mag[k]},
            )
            # Query the region for the current magnitude bin
            fgs_table = v.query_region(
                self.data.body, width=max(sizeTrim[0] * 2, sizeTrim[1] * 2)
            )

            fgs_sources = fgs_table[0]
            fgs_RA, fgs_DEC = (
                jnp.asarray(fgs_sources["RAJ2000"]),
                jnp.asarray(fgs_sources["DEJ2000"]),
            )
            fgs = jnp.vstack([fgs_RA, fgs_DEC]).T

            fgs_unique = [tuple(elem) for elem in fgs if tuple(elem) not in fgs_set]

            # Add unique sources to the set and the list
            for elem in fgs_unique:
                fgs_set.add(elem)

            fgs_list.append(jnp.array(fgs_unique))

        return fgs_list, rad_fac

    def get_mask_source(self) -> jnp.ndarray:
        px_size = read.pixel_size_arcsec(self.data_hdu.header)

        wcs = WCS(self.data_hdu.header)

        position_px = wcs.all_world2pix(
            self.data.geometry.ra, self.data.geometry.dec, 0
        )
        rma_ = self.data.geometry.semiMajorAxis / 2
        rmi_ = rma_ / self.data.geometry.axialRatio

        rma = int(jnp.ceil(rma_ / px_size))
        rmi = int(jnp.ceil(rmi_ / px_size))

        region = """
                image
                ellipse({},{},{},{},{})
                """.format(
            position_px[0],
            position_px[1],
            rma,
            rmi,
            self.data.geometry.positionAngle,
        )

        r = pyregion.parse(region)
        mask_reg = r.get_mask(hdu=self.data_hdu)
        return mask_reg


class ForegroundMaskSinglePass(ForegroundMask):
    pass


class ForegroundMaskMonteCarlo(ForegroundMask):
    pass


class ForegroundMaskAutomaticDifferentiation(ForegroundMask):
    pass
