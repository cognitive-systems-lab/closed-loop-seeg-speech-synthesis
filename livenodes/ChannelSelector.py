from . import Node
import numpy as np


class ChannelSelector(Node.Node):
    def __init__(self, exclude=None, name='ChannelSelector'):
        super(ChannelSelector, self).__init__(name=name)
        self.bad_channels = exclude

    def add_data(self, data_frame, data_id=0):
        cleaned_data_frame = np.delete(data_frame, self.bad_channels, axis=1)
        self.output_data(cleaned_data_frame)
