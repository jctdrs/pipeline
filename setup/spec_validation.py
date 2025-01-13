import yaml
from typing import Any

from setup.pipeline_validation import Pipeline


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


class Specification:
    def __init__(self, spec_path: str):
        self.spec_path = spec_path

    def validate(self) -> Any:
        # Check if specification exists
        try:
            f = open(self.spec_path, "r")
        except OSError:
            print(f"[ERROR] File {self.spec_path} not found.")
            exit()

        # Check if specification is valid YAML
        # In case of failure, capture line/col for debug
        try:
            schema = yaml.load(f, UniqueKeyLoader)
            f.close()
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            line = e.problem_mark.line + 1  # type: ignore
            column = e.problem_mark.column + 1  # type: ignore
            print(f"[ERROR] YAML parsing error at line {line}, column {column}.")
            exit()
        except DuplicateKeyError as e:
            print(f"[ERROR] Duplicate definition of '{e.key}'.")
            exit()

        # JSON is a super-set of YAML therefore we can use a JSON schema
        # validator to validate.
        spec = Pipeline.model_validate(schema)

        return spec
