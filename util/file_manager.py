import yaml
import typing

PIPELINE_STEP_CONFIG: dict = {
    "hip.convolution": {"name", "kernel"},
    "hip.test": {"test"},
    "hip.background": {"cellSize"},
    "hip.cutout": {"raTrim", "decTrim"},
    "hip.reproject": {"target"},
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


class FileManager:
    def __init__(self, spec_path: str):
        self.spec_path = spec_path
        self._parse()

    def _parse(self) -> typing.Any:
        # Check if specification exists
        try:
            f = open(self.spec_path, "r")
        except OSError:
            print(f"[ERROR]\tFile {self.spec_path} not found.")
            exit()

        # Check if specification is valid YAML
        # In case of failure, capture line/col for debug
        try:
            self.spec = yaml.load(f, UniqueKeyLoader)
            f.close()
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as e:
            line = e.problem_mark.line + 1  # type: ignore
            column = e.problem_mark.column + 1  # type: ignore
            print(f"[ERROR]\tYAML parsing error at line {line}, column {column}.")
            exit()
        except DuplicateKeyError as e:
            print(f"[ERROR]\tDuplicate definition of '{e.key}'.")
            exit()

        self.check_yaml_specification()
        return None

    def check_yaml_block(
        self, block_name: str, block: dict, required: typing.Set[str], field_name: str = ""
    ) -> typing.Any:
        if block is None:
            print(f"[ERROR]\tKey '{block_name}' is empty.")
            exit()

        elif not isinstance(block, dict | set):
            print(f"[ERROR]\tField '{field_name}' in '{block_name}' should " "be a list.")
            exit()

        elif not required.issubset(block):
            missing_keys = required.difference(block)
            message = ", ".join(item for item in missing_keys)
            print(f"[ERROR]\tField(s) missing in '{block_name}': {message}.")
            exit()
        return None

    def check_step(self, step: dict) -> typing.Any:
        name: dict = step["step"]
        if name not in PIPELINE_STEP_CONFIG:
            print(f"[ERROR]\tPipeline step '{name}' not defined.")
            exit()

        if "parameters" not in step:
            step["parameters"] = {}
        else:
            pars = step["parameters"]
            if not set(pars.keys()).issubset(PIPELINE_STEP_CONFIG[name]):
                excess_keys = set(pars.keys()).difference(PIPELINE_STEP_CONFIG[name])
                message = ", ".join(item for item in excess_keys)
                print(f"[ERROR]\tExcessive field(s) in '{name}': {message}.")
                exit()

            self.check_yaml_block("parameters", pars, PIPELINE_STEP_CONFIG[name])
        return None

    def check_yaml_specification(self) -> typing.Any:
        required_top: typing.Set[str] = {"config", "data", "pipeline"}
        required_config: typing.Set[str] = {"parallel"}
        required_data: typing.Set[str] = {"body", "bands", "geometry"}
        required_band: typing.Set[str] = {"input", "name", "calError"}
        required_pipeline: typing.Set[str] = {"step"}
        required_geom: typing.Set[str] = {
            "ra",
            "dec",
            "distance",
            "inclination",
            "positionAngle",
            "majorAxis",
            "axialRatio",
            "radius",
            "redshift",
        }

        # Check for top level keys
        self.check_yaml_block("specification file", self.spec, required_top)

        # Check for config keys
        self.config: dict = self.spec["config"]
        if isinstance(self.config, list):
            print("[ERROR]\tElements in 'config' should not be in a list.")
            exit()

        self.check_yaml_block("config", self.config, required_config)

        # Check for data keys
        self.data: dict = self.spec["data"]
        if isinstance(self.data, list):
            print("[ERROR]\tElements in 'data' should not be in a list.")
            exit()

        self.check_yaml_block("data", self.data, required_data)

        if isinstance(self.data["geometry"], list):
            print("[ERROR]\tElements in 'geometry' should not be in a list.")
            exit()

        self.check_yaml_block("geometry", self.data["geometry"], required_geom)

        # Check for every band keys
        for item in self.data["bands"]:
            self.check_yaml_block("bands", item, required_band, "input")

        # Check for pipeline keys
        self.pipeline: dict = self.spec["pipeline"]
        # Check for every step keys
        for item in self.pipeline:
            self.check_yaml_block("pipeline", item, required_pipeline, "step")
            self.check_step(item)
        return None
