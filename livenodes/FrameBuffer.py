import numpy as np
import scipy
import scipy.signal
import multiprocessing

from . import Node


class FrameBuffer(Node.Node):
    """
    Takes a continuous stream of data as input and outputs a stream of frames
    """
    def __init__(self, frame_size_ms, frame_shift_ms, sample_rate, filter_coefficients = None, warm_start = False, name = "FrameBuffer"):
        """Initializes a ring buffer with the neccesary length to filter and frame data.
           Can optionally apply a _causal_ IIR SOS filter to the data before framing. Note that
           while fractional frame shifts are supported, the frame shift has to always be
           at least one full frame."""
        super(FrameBuffer, self).__init__(name = name)
        
        # Make sure no integer accidents happen
        frame_size_ms = float(frame_size_ms)
        frame_shift_ms = float(frame_shift_ms)
        sample_rate = float(sample_rate)
        
        # Frame size and shift
        self.sample_rate = sample_rate
        self.frame_size = int((frame_size_ms / 1000.0) * self.sample_rate)
        self.warm_start = warm_start
        self.next_frame_at = self.frame_size              
        
        # For informational purposes
        self.total_delay = (self.next_frame_at / self.sample_rate) * 1000.0
        
        # Frame tracking
        self.first_frame_at_ms = (float(self.next_frame_at) / self.sample_rate) * 1000.0
        self.frame_count = 0        
        self.buffer_length = max(int(self.next_frame_at * 2.5), 2048)
        
        self.frame_shift_ms = frame_shift_ms
        
        # Data buffer and positions
        self.reset_data_buffers = multiprocessing.Value('b', True)
        
        # Potentially: Filtering
        self.filter_coefficients = None
        if not filter_coefficients is None:
            self.filter_coefficients = np.array(filter_coefficients)
            
        # Return buffer    
        self.return_buffer = None

    def reset_buffer(self):
        """
        Enables buffer resetting in add_data(). Has to be called each time the input process has changed.
        """
        self.reset_data_buffers.value = True



    def add_data(self, data, data_id = None):
        """Add a single frame of data, process it and potentially call callbacks.
           Buffer is allocated on first call."""
        
        # 2Dify, if need be
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        # Allocate buffers, if none
        if self.reset_data_buffers.value == True:
            # Reset output buffers and tracking
            self.data_buffer = []
            self.frame_pos = 0
            self.buffer_pos = 0
            
            # Reset input tracking
            self.next_frame_at = self.frame_size
            self.first_frame_at_ms = (float(self.next_frame_at) / self.sample_rate) * 1000.0
            self.frame_count = 0
            
            # Set up actual buffers
            self.reset_data_buffers.value = False
            self.data_buffer = np.zeros((self.buffer_length, data.shape[1]))
            self.return_buffer = np.zeros((self.frame_size, data.shape[1]))
            
            # Generate filter initial state if needed
            if not self.filter_coefficients is None:
                self.filter_state = scipy.signal.sosfilt_zi(self.filter_coefficients)                
                self.filter_state = np.repeat(self.filter_state, data.shape[1], axis = -1).reshape(
                    self.filter_state.shape[0], self.filter_state.shape[1], data.shape[1])
                if self.warm_start == False:
                    for i in range(data.shape[1]):
                        self.filter_state[:,:,i] *= data[0,i]

            # Zero-fill, if warm start is desired
            if self.warm_start:
                minus_one_shift = self.next_frame_at - int((self.frame_shift_ms / 1000.0) * self.sample_rate)
                assert(minus_one_shift > 0)
                self.add_data(np.zeros((minus_one_shift, data.shape[1])))
        
        #check if input process has changed
        try:
            self.buffer_pos
        except AttributeError:
            raise AttributeError("buffer_pos is not defined. Call reset_buffer() if input process has changed.")
        
        # Add new datum to buffer
        data_step = self.buffer_length - 1
        for idx in range(0, len(data), data_step):
            # Figure out step length
            step_length = min(data_step, len(data) - idx)
            
            # Figure out "output" ranges
            out_start_a = self.buffer_pos
            out_end_a = self.buffer_pos + step_length
            out_start_b = 0
            out_end_b = (self.buffer_pos + step_length) % self.buffer_length

            if out_end_b != out_end_a:
                out_end_a = self.buffer_length
            else:
                out_end_b = 0    
            out_len_a = out_end_a - out_start_a
            out_len_b = out_end_b - out_start_b

            # Figure out "input" ranges
            in_start_a = idx
            in_end_a = in_start_a + out_len_a
            in_start_b = in_end_a
            in_end_b = in_start_b + out_len_b

            # Copy data, possibly filter
            if self.filter_coefficients is None:
                self.data_buffer[out_start_a:out_end_a] = data[in_start_a:in_end_a]
    
                if out_len_b != 0:
                    self.data_buffer[out_start_b:out_end_b] = data[in_start_b:in_end_b]

            else:
                self.data_buffer[out_start_a:out_end_a], self.filter_state = scipy.signal.sosfilt(
                    self.filter_coefficients, data[in_start_a:in_end_a], axis = 0, zi = self.filter_state)
                if out_len_b != 0:
                    self.data_buffer[out_start_b:out_end_b], self.filter_state = scipy.signal.sosfilt(
                        self.filter_coefficients, data[in_start_b:in_end_b], axis = 0, zi = self.filter_state)
                    
            max_frame_pos = self.frame_pos + step_length

            while self.frame_pos < max_frame_pos:

                old_frame_pos = self.frame_pos
                self.frame_pos = min(self.next_frame_at, max_frame_pos)
                
                real_advance_frames = self.frame_pos - old_frame_pos
                self.buffer_pos = (self.buffer_pos + real_advance_frames) % self.buffer_length
                
                
                if self.frame_pos == self.next_frame_at:
                    # Get data for return
                    data_idx = self.buffer_pos
                    data_start_idx = data_idx - self.frame_size
                    
                    # Assign to return buffer, split or proper
                    if data_start_idx >= 0:
                        self.return_buffer = (self.data_buffer[data_start_idx:data_idx]).copy()
                        
                    else:
                        return_buffer_split_idx = abs(data_start_idx)
                        
                        data_start_idx = data_start_idx + self.buffer_length

                        self.return_buffer[0:return_buffer_split_idx] = (self.data_buffer[data_start_idx:])
                        self.return_buffer[return_buffer_split_idx:] = (self.data_buffer[0:data_idx])
                    
                    self.frame_count += 1
                    self.output_data(self.return_buffer)
                    
                    # Advance frame
                    self.next_frame_at = round(((self.first_frame_at_ms + self.frame_count * self.frame_shift_ms) / 1000.0) * self.sample_rate)
