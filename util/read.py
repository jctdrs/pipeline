from typing import Union
from typing import Any
from typing import List

import math

from astropy.io import fits


def pixel_size_arcsec(header: fits.header.Header) -> Union[float, Any]:
    px_size_fits = _check_px_size(header)

    if px_size_fits[2] == "CDELT":
        return abs(px_size_fits[0]) * 3600

    elif px_size_fits[2] == "CD":
        return math.sqrt(px_size_fits[0] ** 2 + px_size_fits[1] ** 2) * 3600

    elif px_size_fits[2] == "PC":
        return math.sqrt(px_size_fits[0] ** 2 + px_size_fits[1] ** 2) * 3600

    else:
        msg = "[ERROR] Unable to get pixel scale from image header."
        raise KeyError(msg)


def _check_px_size(
    header: fits.header.Header,
) -> Union[tuple[float, float, str], Any]:
    keys: List = list(header.keys())
    if ("CDELT1" in keys) and ("CDELT2" in keys):
        if (
            (header["CDELT1"] != 0)
            and (header["CDELT1"] != 1)
            and (header["CDELT1"] != -1)
        ):
            if (
                (header["CDELT2"] != 0)
                and (header["CDELT2"] != 1)
                and (header["CDELT2"] != -1)
            ):
                return (
                    float(header["CDELT1"]),
                    float(header["CDELT2"]),
                    "CDELT",
                )

    elif ("CD1_1" in keys) and ("CD2_2" in keys):
        if (
            (header["CD1_1"] != 0)
            and (header["CD1_1"] != 1)
            and (header["CD1_1"] != -1)
        ):
            if (
                (header["CD2_2"] != 0)
                and (header["CD2_2"] != 1)
                and (header["CD2_2"] != -1)
            ):
                return (float(header["CD1_1"]), float(header["CD2_2"]), "CD")

    elif ("PC1_1" in keys) and ("PC2_2" in keys):
        if (
            (header["PC1_1"] != 0)
            and (header["PC1_1"] != 1)
            and (header["PC1_1"] != -1)
        ):
            if (
                (header["PC2_2"] != 0)
                and (header["PC2_2"] != 1)
                and (header["PC2_2"] != -1)
            ):
                return (float(header["PC1_1"]), float(header["PC2_2"]), "PC")

    else:
        msg = "[ERROR] Unable to get pixel scale from header."
        raise KeyError(msg)

    return None


def shape(
    header: fits.header.Header,
) -> Union[Any, tuple[int, int]]:
    keys: List = list(header.keys())
    if "NAXIS" in keys and header["NAXIS"] == 2 and "NAXIS1" in keys:
        xsize = header["NAXIS1"]
    else:
        msg = "[ERROR] Unable to get number of pixels from header."
        raise KeyError(msg)

    if "NAXIS" in keys and header["NAXIS"] == 2 and "NAXIS2" in keys:
        ysize = header["NAXIS2"]
    else:
        msg = "[ERROR] Unable to get number of pixels from header."
        raise KeyError(msg)

    return (xsize, ysize)


def unit(header: fits.header.Header) -> Union[Any, str]:
    keys: List = list(header.keys())

    if "BUNIT" in keys:
        return header["BUNIT"]
    elif "SIGUNIT" in keys:
        return header["SIGUNIT"]
    elif "ZUNIT" in keys:
        return header["ZUNIT"]
    else:
        msg = "[ERROR] Unable to get unit from header."
        raise KeyError(msg)


def BMAJ(header: fits.header.Header) -> Union[Any, float]:
    keys: List = list(header.keys())

    if "BMAJ" in keys:
        return header["BMAJ"]
    else:
        msg = "[ERROR] Unable to read BMAJ from header."
        raise KeyError(msg)
