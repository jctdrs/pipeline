from typing import List

from astropy.io import fits


def unit(header: fits.header.Header, elem: str) -> None:
    keys: List = List(header.keys())

    if "BUNIT" in keys:
        header["BUNIT"] = elem
    elif "SIGUNIT" in keys:
        header["SIGUNIT"] = elem
    else:
        msg = "[ERROR] Unable to set unit from header."
        raise KeyError(msg)

    return None
