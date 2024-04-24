import math
import typing
from astropy.io import fits


def pixel_scale(header: fits.header.Header) -> float:
    px_size_fits = _check_px_size(header)

    if px_size_fits[2] == "CDELT":
        return abs(px_size_fits[0]) * 3600

    elif px_size_fits[2] == "CD":
        return math.sqrt(px_size_fits[0] ** 2 + px_size_fits[1] ** 2) * 3600

    elif px_size_fits[2] == "PC":
        return math.sqrt(px_size_fits[0] ** 2 + px_size_fits[1] ** 2) * 3600

    else:
        print("[ERROR]\tUnable to get pixel scale from image header.")
        exit()


def _check_px_size(header: fits.header.Header) -> typing.Tuple[float, float, str]:
    keys: list = list(header.keys())
    if ("CDELT1" in keys) and ("CDELT2" in keys):
        if (header["CDELT1"] != 0) and (header["CDELT1"] != 1) and (header["CDELT1"] != -1):
            if (header["CDELT2"] != 0) and (header["CDELT2"] != 1) and (header["CDELT2"] != -1):
                return (header["CDELT1"], header["CDELT2"], "CDELT")

    elif ("CD1_1" in keys) and ("CD2_2" in keys):
        if (header["CD1_1"] != 0) and (header["CD1_1"] != 1) and (header["CD1_1"] != -1):
            if (header["CD2_2"] != 0) and (header["CD2_2"] != 1) and (header["CD2_2"] != -1):
                return (header["CD1_1"], header["CD2_2"], "CD")

    elif ("PC1_1" in keys) and ("PC2_2" in keys):
        if (header["PC1_1"] != 0) and (header["PC1_1"] != 1) and (header["PC1_1"] != -1):
            if (header["PC2_2"] != 0) and (header["PC2_2"] != 1) and (header["PC2_2"] != -1):
                return (header["PC1_1"], header["PC2_2"], "PC")

    else:
        print("[ERROR]\tUnable to get pixel scale from header.")
        exit()


def shape(header: fits.header.Header) -> typing.Tuple[int, int]:
    keys: list = list(header.keys())
    if "NAXIS" in keys and header["NAXIS"] == 2 and "NAXIS1" in keys:
        xsize = header["NAXIS1"]
    else:
        print("[ERROR]\tUnable to get number of pixels from header.")
        exit()

    if "NAXIS" in keys and header["NAXIS"] == 2 and "NAXIS2" in keys:
        ysize = header["NAXIS2"]
    else:
        print("[ERROR]\tUnable to get number of pixels from header.")
        exit()

    return (xsize, ysize)
