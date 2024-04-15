from util import pipeline


class Convolution:
    def __init__(self, band, target):
        self.band = band
        self.target = target

    def run(self):
        print(f"[INFO]\tConvoluting {self.band['input']} to {self.target}")
        return

    def kernel_conv(self):
        """ Perform the kernel convolution on the provided data """
        hdr = header.copy()
        kernel_file = get_kernel(band, target_band, header)
        if band == target_band:
            return data, hdr

        # check if path exists
        if not path.exists(path_kernel + kernel_file):
            print(f"[Error] {kernel_file} not found. No convolution performed")
            return data, hdr

        # if file exists, we can extract it
        kernel, kernel_hdr = fits.getdata(path_kernel + kernel_file, header=True)

        # do the actual convolution
        data_conv, kernel_out = ca.do_the_convolution(data, hdr, kernel, kernel_hdr)

        # update the header with the new beam size
        hdr.set("BMAJ", BMAJ / 3600, "[deg] Beam major axis in degrees")
        hdr.set("BMIN", BMIN / 3600, "[deg] Beam minor axis in degrees")

        return data_conv, hdr
