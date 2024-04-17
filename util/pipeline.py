import multiprocessing
import abc

from hip import convolution
from util import file_manager

Interface: dict = {"hip.convolution": convolution.Convolution}


class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        self.file_mng = file_mng
        self.result: list = []

    @classmethod
    def create(cls, file_mng: file_manager.FileManager):
        parallel: bool = file_mng.config["parallel"]
        if parallel:
            return PipelineParallel(file_mng)
        else:
            return PipelineSequential(file_mng)

    @abc.abstractmethod
    def execute(self):
        pass


class PipelineSequential(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        super().__init__(file_mng)

    def execute(self):
        bands: dict = self.file_mng.data["bands"]
        for idx, band in enumerate(bands):
            self.result.append(None)
            self._target(band, idx)
        return

    def _target(self, band: dict, idx: int):
        for task in self.file_mng.pipeline:
            s = Interface[task["step"]](band, **task["parameters"])
            data = s.run()
            self.result[idx] = data
        return


class PipelineParallel(Pipeline):
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        super().__init__(file_mng)
        self.processes: list = []

    def execute(self):
        # Spawn a process for every set of bands
        bands = self.file_mng.data["bands"]
        for idx, band in enumerate(bands):
            self.result.append(multiprocessing.Queue())
            p = multiprocessing.Process(
                target=self._target,
                args=(
                    band,
                    idx,
                ),
            )
            self.processes.append(p)
            p.start()

        for process in self.processes:
            process.join()

        return [queue.get() for queue in self.result]

    def _target(self, band: dict, idx: int):
        for task in self.file_mng.pipeline:
            s = Interface[task["step"]](band, **task["parameters"])
            data = s.run()
            self.result[idx].put(data)
        return
