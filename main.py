import argparse

from util import file_manager
from util import pipeline

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, help="Specification YAML file", required=True
    )
    args = parser.parse_args()
    
    file_mng = file_manager.FileManager(args.file)
    file_mng.parse()
    
    pipe = pipeline.Pipeline(file_mng)
    pipe.execute()
