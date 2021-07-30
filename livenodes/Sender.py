import time
import numpy as np
from multiprocessing import Process
import gc
from . import Node


class Sender(Node.Node):
    """
    Takes a numpy-Array as Input and streams it.
    """
    def __init__(self, data, sample_rate, frame_size_ms, asap=False, name='Sender'):
        """
        Initialize a Sender.
        """
        super(Sender, self).__init__(has_inputs=False, name=name)
        self.sample_rate = sample_rate
        self.data = data
        self.frame_size_ms = frame_size_ms
        self.asap = asap
        self.feeder_process = None
        
    def sender_process(self):
        """
        Streams the data and calls frame callbacks for each frame.
        """
        samples_per_frame = int(self.sample_rate / 1000 * self.frame_size_ms)
        time_val = time.time()
        time_val_init = time_val
        for sample_cnt in range(0, len(self.data), samples_per_frame):
            samples = self.data[sample_cnt:sample_cnt+samples_per_frame]
            if not self.asap:
                while time.time() - time_val < (1.0 / 1000.0) * self.frame_size_ms:
                    time.sleep(0.000001)
                time_val = time_val_init + sample_cnt / self.sample_rate
            self.output_data(np.array(samples))
    
    def send_new(self, data):
        """
        Stream a new data array. Must wait for completion before doing this.
        """
        if self.feeder_process is None:
            gc.collect()
            self.data = data
            self.feeder_process = Process(target = self.sender_process)
            self.feeder_process.start()
        super(Sender, self).start_processing()
        
    def wait_for_completion(self):
        """
        Waits for all data to have been sent
        """
        if self.feeder_process is not None:
            self.feeder_process.join()
        self.feeder_process = None
        
    def start_processing(self, recurse=True):
        """
        Starts the streaming process.
        """
        if self.feeder_process is None and self.data is not None:
            self.feeder_process = Process(target=self.sender_process)
            self.feeder_process.start()
        super(Sender, self).start_processing(recurse)
        
    def stop_processing(self, recurse=True):
        """
        Stops the streaming process.
        """
        super(Sender, self).stop_processing(recurse)
        if self.feeder_process is not None:
            self.feeder_process.terminate()
        self.feeder_process = None
