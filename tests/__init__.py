import sys
from pathlib import Path

# Add the parent directory to the path so the tests can
# import all the source code
sys.path.append(str(Path(__file__).resolve().parent.parent.joinpath("pipeline")))
