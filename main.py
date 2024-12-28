import argparse

from setup import spec_validation
from setup import pipeline


def main():
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

    spec = spec_validation.Specification(spec_path).validate()

    pipe = pipeline.Pipeline.create(spec)
    pipe.execute()

    return


if __name__ == "__main__":
    main()
