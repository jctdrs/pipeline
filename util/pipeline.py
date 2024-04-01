from hip import convolution
from util import file_manager

Interface = {"hip.convolution": convolution.Convolution}


class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        self.file_mng = file_mng
        return

    def execute(self):
        for process in self.pipeline:
            step_routine = Interface[process["step"]]
            data = step_routine(**process["parameters"])
        return data
