import pyaudio
import multiprocessing
import numpy as np
from . import Node
from . import LambdaNode
from . import FrameBuffer
import logging

logger = logging.getLogger('PyAudioSink.py')


class PyAudioSink(Node.Node):
    """
    Multithreaded pyaudio sink.
    """

    def __init__(self, orig_sample_rate, block_size, stereo_channel='both', name='PyAudioSink'):

        # Set up variables
        super(PyAudioSink, self).__init__(name=name)
        self.sample_rate = orig_sample_rate
        self.block_size = block_size
        self.stream = None

        # Specify output channel
        if stereo_channel == 'both':
            self.channels = 1
            self.ch_index = 0
        elif stereo_channel == 'left':
            self.channels = 2
            self.ch_index = 0
        elif stereo_channel == 'right':
            self.channels = 2
            self.ch_index = 1
        else:
            logger.warning('Unknown argument for stereo_channel [{}]. Falling back to both.'.format(stereo_channel))
            self.channels = 1
            self.ch_index = 0

        self.samplePipeOut, self.samplePipeIn = multiprocessing.Pipe(False)
        self.pipe_fill = multiprocessing.Value('i', 0)

        self.block_buffer = FrameBuffer.FrameBuffer(self.block_size, self.block_size, 1000, name=name + '.Block_Buffer')
        self.byte_converter = LambdaNode.LambdaNode(self.int16_converter, name=name + '.Int16')(self.block_buffer)

    def add_data(self, data_frame, data_id=0):
        self.block_buffer.add_data(data_frame)

    def int16_converter(self, data):
        """
        Rescale audio data to match the int16 values
        """
        data = data.flatten()
        mono = np.zeros((len(data), self.channels), dtype=np.int16)
        mono[:, self.ch_index] = data
        self.samplePipeIn.send(mono)

        with self.pipe_fill.get_lock():
            self.pipe_fill.value += 1

    def start_processing(self, recurse=True):
        py_aud = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, status):
            data = self.samplePipeOut.recv()

            with self.pipe_fill.get_lock():
                self.pipe_fill.value -= 1

            return data, pyaudio.paContinue

        # open stream using callback
        self.stream = py_aud.open(format=pyaudio.paInt16, channels=self.channels, rate=self.sample_rate,
                                  frames_per_buffer=self.block_size, output=True, start=False,
                                  stream_callback=callback)

        # wait for first few samples
        while True:
            self.samplePipeOut.poll(None)
            with self.pipe_fill.get_lock():
                if self.pipe_fill.value >= 2:
                    break

        self.stream.start_stream()

        super(PyAudioSink, self).start_processing(recurse)

    def stop_processing(self, recurse=True):
        super(PyAudioSink, self).stop_processing(recurse)
        self.stream.stop_stream()
        self.stream.close()
