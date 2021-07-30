import time
import numpy as np
import multiprocessing
from . import Node


class Receiver(Node.Node):
    """
    Receives data sent by LiveSystemReceiver.
    """
    def __init__(self, perform_timing=False, dont_time=False, name='Receiver'):
        """ 
        Initialize a Receiver.
        """
        super(Receiver, self).__init__(has_outputs = False, dont_time = dont_time, name = name)
        manager = multiprocessing.Manager()
        self.data = manager.list([])
        self.perform_timing = perform_timing
        
    def add_data(self, sample, data_id=None):
        """
        Add a single frame of data
        """
        if self.perform_timing:
            self.data.append([time.time(), sample])
        else:
            self.data.append(sample)
    
    def get_data(self, clear = False):
        """
        Returns the data as a list.
        """
        ret_data = []
        for sample in self.data:
            ret_data.append(sample)
        
        if clear:
            self.data[:] = []
            
        return ret_data
