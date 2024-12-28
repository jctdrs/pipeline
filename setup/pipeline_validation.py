from setup import config_validation
from setup import data_validation
from setup import parameters_validation

from pydantic import BaseModel, model_validator
from typing_extensions import Optional, List


class PipelineStep(BaseModel):
    step: str
    parameters: BaseModel
    diagnosis: Optional[bool] = False

    @model_validator(mode="before")
    def pass_step_to_parameters(self):
        self["parameters"] = parameters_validation.factory_method(
            self["step"], **self["parameters"]
        )
        return self

    @model_validator(mode="after")
    def validate_pipeline_steps(self):
        hip_possible_steps = {
            "hip.convolution",
            "hip.skySubtract",
            "hip.cutout",
            "hip.reproject",
            "hip.integrate",
            "hip.foregroundMasking",
            "hip.test",
        }

        if self.step not in hip_possible_steps:
            msg = f"[ERROR] Step '{self.step}' incorrect."
            raise ValueError(msg)
        return self


class Meta(BaseModel):
    # Individual fields in 'Meta' are not mandatory. In case they are not
    # defined by the user, then they are manually filled with default values.
    name: str = "Default"
    author: str = "Default"
    version: str = "Default"
    description: str = "Default"


class Pipeline(BaseModel):
    # The whole 'meta' section is not mandatory to define. In case it is not
    # defined by the user, then it is manually filled with all default values.
    meta: Optional[Meta] = Meta(
        name="Default",
        author="Default",
        version="Default",
        description="Default",
    )
    config: config_validation.Config
    data: data_validation.Data
    pipeline: List[PipelineStep]


def test():
    pipeline = {
        "meta": {
            "name": "HIP",
            "author": "JT",
            "version": "1.0",
            "description": "This runs HIP",
        },
        "config": {"mode": "Single Pass", "niter": 1},
        "data": {
            "body": "NGC4254",
            "geometry": {
                "distance": 12.4,
                "redshift": 123.4,
                # "ra": 123.2,
                # "dec": 1234,
                # "positionAngle": 54,
                # "axialRatio": 124,
                # "semiMajorAxis": 123,
                # "inclination": 123,
                # "radius": 123,
            },
            "bands": [
                {
                    "input": "/home/jtedros/Repo/pipeline/data/inputs/NGC4254/NGC4254_PACS1.fits",
                    # "error": "/home/jtedros/Repo/pipeline/data/inputs/NGC4254/NGC4254_PACS1.fits",
                    "output": "/home/jtedros/Repo/pipeline/data/inputs/NGC4254/NGC4254_PACS1.fits",
                    "name": "PACS1",
                    "calError": 5.3,
                },
            ],
        },
        "pipeline": [
            {
                "step": "hip.skySubtract",
                "diagnosis": True,
                "parameters": {"cellSize": 13},
            },
            {
                "step": "hip.cutout",
                "diagnosis": True,
                "parameters": {"raTrim": 10, "decTrim": 10},
            },
            {
                "step": "hip.convolution",
                "diagnosis": True,
                "parameters": {
                    "kernel": "/home/jtedros/Repo/pipeline/data/kernels/Kernel_LowRes_PACS1_to_SPIRE3.fits"
                },
            },
            {
                "step": "hip.reproject",
                "diagnosis": False,
                "parameters": {
                    "target": "/home/jtedros/Repo/pipeline/data/inputs/NGC4254/NGC4254_SPIRE3.fits"
                },
            },
        ],
    }

    a = Pipeline.model_validate(pipeline)
    print(a.data.bands[0].error)
