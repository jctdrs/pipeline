import argparse
import warnings

from packages import pipeline
from models import spec

from pydantic import ValidationError

warnings.filterwarnings("ignore")


def main() -> None:
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Specification YAML file",
        required=True,
    )
    args = parser.parse_args()
    spec_path: str = args.file

    try:
        spc = spec.Specification(spec_path).validate()
    except ValidationError as e:
        print(e)
        exit(-1)

    pipe = pipeline.PipelineGeneric.create(spc)
    pipe.execute()

    return None


if __name__ == "__main__":
    main()
