from livenodes import Node
import numpy as np
from scipy.ndimage import gaussian_filter


class Dequantization(Node.Node):
    """
    Perform the de-quantization routine to get back a spectrogram
    """
    def __init__(self, medians_array, name='ChannelSelector'):
        super(Dequantization, self).__init__(name=name)
        self.medians_array = medians_array
        self.c = np.arange(len(self.medians_array))

    def add_data(self, data_frame, data_id=0):
        spec_vector = self.medians_array[self.c, data_frame.astype(int)]
        spec_smooth = gaussian_filter(spec_vector, sigma=0.5)
        self.output_data(spec_smooth)
