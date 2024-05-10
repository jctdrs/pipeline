import typing
import math

from util import read

import astropy
from astropy.wcs import WCS
from photutils.aperture.ellipse import EllipticalAperture
from photutils.aperture import aperture_photometry
import numpy as np
import numpy.ma as ma


class Integrate:
    def __init__(
        self,
        data_hdu: astropy.io.fits.hdu.image.PrimaryHDU,
        err_hdu: typing.Union[astropy.io.fits.hdu.image.PrimaryHDU, typing.Any],
        geom: dict,
        instruments: dict,
    ):
        self.data_hdu = data_hdu
        self.err_hdu = err_hdu
        self.geom = geom
        self.instruments = instruments

    def run(self):
        wcs = WCS(self.data_hdu.header)
        px_size = abs(read.pixel_size(self.data_hdu.header)[0] * 3600)
        ra_ = self.geom["ra"]
        dec_ = self.geom["dec"]
        rma_ = self.geom["majorAxis"]
        rmi_ = rma_ / self.geom["axialRatio"]
        position_px = wcs.all_world2pix(ra_, dec_, 0)
        rma = int(math.ceil(rma_ / px_size))
        rmi = int(math.ceil(rmi_ / px_size))

        nan = ma.masked_invalid(self.data_hdu.data)

        a = [
            int(rma * 0.2),
            int(rma * 0.6),
            int(rma * 1.0),
            int(rma * 1.3),
            int(rma * 1.7),
            int(rma * 2.0),
            int(rma * 3.0),
            int(rma * 3.5),
            int(rma * 4.0),
            int(rma * 4.5),
            int(rma * 5.0),
            int(rma * 5.5),
            int(rma * 6.0),
            int(rma * 6.5),
            int(rma * 7.0),
            int(rma * 7.5),
            int(rma * 8.0),
            int(rma * 8.5),
            int(rma * 9.0),
            int(rma * 9.5),
            int(rma * 10.0),
        ]

        b = [
            int(rma * 0.2),
            int(rma * 0.6),
            int(rma * 1.0),
            int(rma * 1.3),
            int(rma * 1.7),
            int(rma * 2.0),
            int(rmi * 2.5),
            int(rmi * 3.0),
            int(rmi * 3.5),
            int(rmi * 4.0),
            int(rmi * 4.5),
            int(rmi * 5.0),
            int(rma * 5.5),
            int(rma * 6.0),
            int(rma * 6.5),
            int(rma * 7.0),
            int(rmi * 7.5),
            int(rmi * 8.0),
            int(rmi * 8.5),
            int(rmi * 9.0),
            int(rmi * 9.5),
            int(rmi * 10.0),
        ]

        apertures = [
            EllipticalAperture(position_px, a=ai + 1e-3, b=bi + 1e-3, theta=self.geom["positionAngle"])
            for (ai, bi) in zip(a, b)
        ]
        phot_table = aperture_photometry(nan.data, apertures, error=None, mask=nan.mask)

        for col in phot_table.colnames:
            phot_table[col].info.format = "%.8g"

        integrated_flux_list = [
            phot_table["aperture_sum_0"],
            phot_table["aperture_sum_1"],
            phot_table["aperture_sum_2"],
            phot_table["aperture_sum_3"],
            phot_table["aperture_sum_4"],
            phot_table["aperture_sum_5"],
        ]

        integrated_flux = np.array(integrated_flux_list[0:])
        # integrated_L = (integrated_flux * 1e-26) * pow((self.geom["distance"] * 3.086e22), 2) / (3.846e26)

        return integrated_flux
