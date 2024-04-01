import yaml
import os

class DuplicateKeyError(Exception):
    def __init__(self, key):
        self.key = key

class UniqueKeyLoader(yaml.SafeLoader):
     def construct_mapping(self, node, deep=False):
         mapping = []
         for key_node, value_node in node.value:
             key = self.construct_object(key_node, deep=deep)
             if key in mapping:
                raise DuplicateKeyError(key)
             mapping.append(key)
         return super().construct_mapping(node, deep)

class FileManager:
    def __init__(self, spec_path):
        self.spec_path = spec_path

    def parse(self):
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
            line = e.problem_mark.line + 1
            column = e.problem_mark.column + 1
            print(
                f"[ERROR]\tYAML parsing error at line {line}, column {column}."
            )
            exit()
        except DuplicateKeyError as e:
            print(f"[ERROR]\tDuplicate definition of \'{e.key}\'.")
            print(f"[HINT]\tField \'{e.key}\' might be a list. " 
                  "Should be declared with '-'.")
            exit()

        self.check_yaml_specification(self.spec)
        self.check_file(self.spec)

    def check_yaml_block(
        self, block_name: str, block: dict, required: dict, field_name: str = ""
    ):
        # Check that the required keys are in the specification
        if not isinstance(block, dict):
            print(
                f"[ERROR]\tField \'{field_name}\' in \'{block_name}\' should "
                "be a list."
            )
            print(f"[HINT]\tLists are declared with a '-'.")
            exit()

        elif not required.issubset(block):
            missing_keys = set(required).difference(block)
            message = ", ".join(item for item in missing_keys)
            print(f"[ERROR]\tField(s) in {field_name} missing: {message}.")
            exit()

    def check_yaml_specification(self, spec: dict) -> None:
        required_top: dict = {"config", "data", "pipeline"}
        required_config: dict = {"input_dir", "output_dir", "kernel_dir"}
        required_data: dict = {"body", "bands"}
        required_band: dict = {"input"}
        required_pipeline: dict = {"step"}

        # Check for top level keys
        self.check_yaml_block("'specification'", spec, required_top)

        # Check for config
        config = spec["config"]
        if isinstance(config, list):
            print("[ERROR]\tFields in 'config' should be elements.") 
            print("[HINT]\tElements are declared without a '-'.")
            exit()
        
        self.check_yaml_block("'config'", config, required_config)

        # Check for data
        data = spec["data"]
        # Check for every body
        for item in data:
            self.check_yaml_block("data", item, required_data, "body")
            # Check for every band
            for subitem in item["bands"]:
                self.check_yaml_block(
                    "bands", subitem, required_band, "input"
                )

        # Check for pipeline
        pipe = spec["pipeline"]
        # Check for every step
        for item in pipe:
            self.check_yaml_block(
                "pipeline", item, required_pipeline, "step"
            )

        return

    def check_file(self, spec: dict) -> None:
        bodies: dict = spec["data"]
        inp_dir: str = spec["config"]["input_dir"]
        status: bool = True

        files_checked = []
        self.files_without_error = []
        # Check that files in specification exists for all
        # bodies, all bands, and all input and error files
        for body in bodies:
            for band in body["bands"]:
                if "error" not in band.keys():
                    self.files_without_error.append(f"{inp_dir}/{band['input']}")
                for key in band.keys():
                    filename: str = f"{inp_dir}/{band[key]}"
                    if filename not in files_checked:
                        if os.path.exists(filename):
                            files_checked.append(filename)
                        else:
                            print(f"[ERROR]\tFile '{filename}' not found.")
                            status = status and False
                    else:
                        pass
