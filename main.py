import argparse
import time
import contextlib
import warnings

from util import file_manager
from util import pipeline
from util import setup_manager


@contextlib.contextmanager
def chrono():
    start = time.time()
    yield
    end = time.time()
    print(f"[DEBUG]\tElapsed time {end-start:.3f} seconds.")


def main() -> pipeline.Pipeline:
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="Specification YAML file", required=True)
    args = parser.parse_args()
    spec_path: str = args.file

    file_mng = file_manager.FileManager(spec_path)
    setup_manager.SetupManager(file_mng).set()

    pipe = pipeline.Pipeline.create(file_mng)
    pipe.execute()
    return pipe


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pipe = main()
