import typing
import math
import copy


from astroquery.vizier import Vizier

from astropy.wcs import WCS
import astropy.units as au
import astropy

# TODO: Remove numpy from here
import numpy as jnp

import pyregion


from util import read


class ForegroundSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def run(self, *args, **kwargs):
        pass


class Foreground(ForegroundSingleton):
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        output_path: str,
        name: str,
        body: str,
        geom: dict,
        instruments: dict,
        diagnosis: bool,
        MC_diagnosis: bool,
        differentiate: bool,
        factor: float,
        raTrim: float,
        decTrim: float,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.output_path = output_path
        self.name = name
        self.body = body
        self.geom = geom
        self.instruments = instruments
        self.diagnosis = diagnosis
        self.MC_diagnosis = MC_diagnosis
        self.differentiate = differentiate
        self.factor = factor
        self.raTrim = raTrim
        self.decTrim = decTrim

    def run(
        self,
    ) -> typing.Tuple[
        astropy.io.fits.hdu.image.PrimaryHDU,
        astropy.io.fits.hdu.image.PrimaryHDU,
        typing.Union[jnp.array, typing.Any],
    ]:
        # generate copy to be masked
        px_size = read.pixel_size_arcsec(self.data_hdu.header)

        # iterate over the fgs and mask them
        wcs = WCS(self.data_hdu.header)

        fgs_list, rad_fac = self.find_fgs()

        # Exlude the central regions of the galaxy from the subtraction
        mask_gal_reg = self.get_mask_source()

        for k in range(len(fgs_list)):
            # convert wcs to pixel
            fgs_pos_px = wcs.all_world2pix(fgs_list[k], 0)
            r_mask = (
                rad_fac[k]
                * self.factor
                * self.instruments[self.name]["RESOLUTION"]["VALUE"]
            ) // 2

            # iterate over the indicidual points
            for i in range(len(fgs_pos_px)):
                x = jnp.ceil(fgs_pos_px[i][0]).astype(int)
                y = jnp.ceil(fgs_pos_px[i][1]).astype(int)

                if (
                    x < self.data_hdu.header["NAXIS1"]
                    and y < self.data_hdu.header["NAXIS2"]
                ):
                    if not mask_gal_reg[y, x]:
                        r_circle = int(jnp.ceil(r_mask / px_size))
                        region = """
                             image
                             circle({},{},{})
                             """.format(fgs_pos_px[i][0], fgs_pos_px[i][1], r_circle)
                        r = pyregion.parse(region)
                        mask_fgs_reg = r.get_mask(hdu=self.data_hdu)

                        self.data_hdu.data[mask_fgs_reg] = jnp.nan

        return self.data_hdu, self.err_hdu, None

    def find_fgs(self):
        mag = ["<13.5", "<14.", "<15.5", "<16.", "<18.", "<40."]
        rad_fac = [4.6, 3.0, 2.1, 1.4, 1.15, 0.7]
        fgs1, fgs2, fgs3, fgs4, fgs5, fgs6 = [], [], [], [], [], []
        fgs_list = [fgs1, fgs2, fgs3, fgs4, fgs5, fgs6]

        sizeTrim = (self.decTrim * au.arcmin, self.raTrim * au.arcmin)

        for k in range(len(fgs_list)):
            # Vizier query for point sources (Gaia Data Release 3 catalog)
            v = Vizier(
                columns=["RAJ2000", "DEJ2000"],
                catalog="I/355",
                row_limit=10000,
                column_filters={"Gmag": mag[k]},
            )
            # look for point sources over a 4 times area with respect the cutout
            # (2*sizeTrim)
            fgs_table = v.query_region(
                self.body, width=max(sizeTrim[0] * 2, sizeTrim[1] * 2)
            )

            fgs_sources = fgs_table[0]
            fgs_skycoord = [fgs_sources["RAJ2000"], fgs_sources["DEJ2000"]]
            fgs_RA = jnp.asarray(fgs_skycoord[0:][0])
            fgs_DEC = jnp.asarray(fgs_skycoord[1:][0])
            fgs = jnp.vstack([fgs_RA, fgs_DEC]).T

            if k == 0:
                fgs_list[k] = copy.deepcopy(fgs)

            else:
                fgs_temp = jnp.empty((0, 2))
                for j in range(k):
                    fgs_temp = jnp.append(fgs_temp, fgs_list[j], axis=0)
                fgs_k = jnp.empty((0, 2))
                for elem in fgs:
                    if elem not in fgs_temp:
                        fgs_k = jnp.append(
                            fgs_k, jnp.array([[elem[0], elem[1]]]), axis=0
                        )
                fgs_list[k] = copy.deepcopy(fgs_k)

        return fgs_list, rad_fac

    def get_mask_source(self):
        px_size = read.pixel_size_arcsec(self.data_hdu.header)

        wcs = WCS(self.data_hdu.header)

        pos_ctr = jnp.vstack([(self.geom["ra"]), self.geom["dec"]]).T
        pos_center_px = wcs.all_world2pix(pos_ctr, 0)
        rma_gal = self.geom["semiMajorAxis"] / 2
        rmi_gal = rma_gal / self.geom["axialRatio"]
        rma_gal_px = math.ceil(rma_gal / px_size)
        rmi_gal_px = math.ceil(rmi_gal / px_size)

        region = """
                image
                ellipse({},{},{},{},{})
                """.format(
            pos_center_px[0][0],
            pos_center_px[0][1],
            rma_gal_px,
            rmi_gal_px,
            self.geom["positionAngle"],
        )

        r = pyregion.parse(region)
        mask_reg = r.get_mask(hdu=self.data_hdu)

        return mask_reg
