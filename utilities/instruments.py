from dataclasses import dataclass


@dataclass
class Resolution:
    WISE1: float = 6.1
    WISE2: float = 6.4
    WISE3: float = 6.5
    WISE4: float = 2

    IRAC1: float = 1.66
    IRAC2: float = 1.72
    IRAC3: float = 1.88
    IRAC4: float = 1.98

    MIPS1: float = 6
    MIPS2: float = 18
    MIPS3: float = 35.2

    PACS1: float = 9
    PACS2: float = 10
    PACS3: float = 13

    SPIRE1: float = 18
    SPIRE2: float = 25
    SPIRE3: float = 36

    HFI1: float = 300
    HFI2: float = 300
    HFI3: float = 300
    HFI4: float = 330
    HFI5: float = 438

    NIKA2_1: float = 12
    NIKA2_2: float = 18

    GALEX_FUV: float = 4.3
    GALEX_NUV: float = 5.3
