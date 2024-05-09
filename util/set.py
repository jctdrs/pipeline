from astropy.io import fits


def unit(header: fits.header.Header, elem: str) -> None:
    keys: list = list(header.keys())

    if "BUNIT" in keys:
        header["BUNIT"] = elem
    elif "SIGUNIT" in keys:
        header["SIGUNIT"] = elem
    else:
        print("[ERROR]\tUnable to set unit from header.")
        exit()
