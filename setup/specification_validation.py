import yaml

# TODO: Remove typing. Go over all typing conventions and take care of python version
import typing

from setup import pipeline_validation

PIPELINE_STEP_CONFIG: dict = {
    "hip.convolution": {"kernel"},
    "hip.background": {"cellSize"},
    "util.cutout": {"raTrim", "decTrim"},
    "hip.reproject": {"target"},
    "util.integrate": {"radius", "calError"},
    "hip.foreground": {"factor", "raTrim", "decTrim"},
    "util.test": {},
}


class DuplicateKeyError(Exception):
    def __init__(self, key):
        self.key = key


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep: bool = False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise DuplicateKeyError(key)
            mapping.append(key)
        return super().construct_mapping(node, deep)


class SpecificationValidation:
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self._parse()

    def _parse(self) -> typing.Any:
        # Check if specification exists
        try:
            f = open(self.spec_path, "r")
        except OSError:
            print(f"[ERROR] File {self.spec_path} not found.")
            exit()

        # Check if specification is valid YAML
        # In case of failure, capture line/col for debug
        try:
            self.spec = yaml.load(f, UniqueKeyLoader)
            f.close()
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            line = e.problem_mark.line + 1  # type: ignore
            column = e.problem_mark.column + 1  # type: ignore
            print(f"[ERROR] YAML parsing error at line {line}, column {column}.")
            exit()
        except DuplicateKeyError as e:
            print(f"[ERROR] Duplicate definition of '{e.key}'.")
            exit()

        pipeline_validation.PipelineValidation.model_validate(self.spec)

        return None
