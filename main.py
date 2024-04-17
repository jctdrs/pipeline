import argparse

from util import file_manager
from util import pipeline


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Specification YAML file", required=True)
    args = parser.parse_args()

    spec_path: str = args.file
    file_mng: file_manager.FileManager = file_manager.FileManager(spec_path)
    pipe: pipeline.Pipeline = pipeline.Pipeline.create(file_mng)

    pipe.execute()


if __name__ == "__main__":
    main()
