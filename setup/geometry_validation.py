import csv

from pydantic import BaseModel, model_validator
from typing import Optional


DUSTPEDIA_APERTURE_PHOTOMETRY = (
    "/home/jtedros/Repo/pipeline/data/config/DustPedia_Aperture_Photometry_2.2.csv"
)
DUSTPEDIA_HYPERLEDA_HERSCHEL = (
    "/home/jtedros/Repo/pipeline/data/config/DustPedia_HyperLEDA_Herschel.csv"
)


class Geometry(BaseModel):
    distance: float
    redshift: float

    # The 'body' attribute is passed from the 'Data' class to the 'Geometry'
    # class through the '@model_validator'. This is necessary in order to
    # parse the corresponding geometry fields in case they are missing.
    body: str

    # Some geometry fields are automatically parsed from the Dustpedia and
    # and photometry datasets located in 'data/config/' in case not defined
    # by the user. In case they are not defined, then their default value
    # is None therefore the use of the 'Optional[float]' type hint.
    ra: Optional[float] = None
    dec: Optional[float] = None
    positionAngle: Optional[float] = None
    axialRatio: Optional[float] = None
    semiMajorAxis: Optional[float] = None
    inclination: Optional[float] = None
    radius: Optional[float] = None

    @model_validator(mode="after")
    def geometry_from_config(self):
        # These are the fields that might not be defined by the user and are
        # therefore parsed from the csv files in the configuration. The
        # dictionary act as interfaces due to the different nomenclature.
        required_phot = {
            "ra": "ra",
            "dec": "dec",
            "positionAngle": "pos_angle",
            "axialRatio": "axial_ratio",
            "semiMajorAxis": "semimaj_arcsec",
        }
        required_dist = {
            "inclination": "incl",
            "radius": "d25",
        }

        # The 'Geometry' fields are scattered into two separate config csv
        # files. In order to avoid opening and parsing the files when not
        # needed.
        none_attrs: list = [key for key, value in vars(self).items() if value is None]
        none_phot: list = [key for key in required_phot if key in none_attrs]
        none_dist: list = [key for key in required_dist if key in none_attrs]

        if none_phot:
            self.parse_csv(DUSTPEDIA_APERTURE_PHOTOMETRY, none_phot, required_phot)

        if none_dist:
            self.parse_csv(DUSTPEDIA_HYPERLEDA_HERSCHEL, none_dist, required_dist)

        return self

    def parse_csv(self, config_path, none_fields, required_fields):
        with open(config_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # The name of the objects have different keys in the csv files.
                if (
                    row.get("name", "") == self.body
                    or row.get("objname", "") == self.body
                ):
                    for key, value in required_fields.items():
                        if key in none_fields:
                            setattr(self, key, float(row[value]))
                    return

        msg = f"[ERROR] Body {self.body} not found."
        raise ValueError(msg)
