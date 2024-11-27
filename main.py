import argparse
import setproctitle

from setup import file_manager
from setup import pipeline
from setup import setup_manager


def main() -> pipeline.Pipeline:
    setproctitle.setproctitle("pipeline")

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, help="Specification YAML file", required=True
    )
    args = parser.parse_args()
    spec_path: str = args.file

    file_mng = file_manager.FileManager(spec_path)
    setup_manager.SetupManager(file_mng).set()

    pipe = pipeline.Pipeline.create(file_mng)
    pipe.execute()

    return pipe


if __name__ == "__main__":
    main()
