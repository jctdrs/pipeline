import multiprocessing

from hip import convolution
from util import file_manager

Interface = {"hip.convolution": convolution.Convolution}

class Pipeline:
    def __init__(self, file_mng: file_manager.FileManager) -> None:
        self.file_mng = file_mng
        self.processes: list  = []
        self.return_queue: list = []

    def start(self, band, idx):
        for task  in self.file_mng.pipeline:
            s = Interface[task["step"]](band, **task["parameters"])
            data = s.run()
            self.return_queue[idx].put(data)

    def execute(self):
        # Spawn a process for every set of bands 
        bands = self.file_mng.data["bands"]
        for idx, band in enumerate(bands):
            self.return_queue.append(multiprocessing.Queue())
            p = multiprocessing.Process(target=self.start, args=(band,idx,))
            self.processes.append(p)
            p.start()

        for process in self.processes:
            p.join()
        
        return [queue.get() for queue in self.return_queue] 
