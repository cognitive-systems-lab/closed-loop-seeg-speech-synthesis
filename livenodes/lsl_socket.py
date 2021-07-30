import numpy as np
import multiprocessing
import logging
from pylsl import StreamInlet, resolve_stream, FOREVER
from . import Node


logger = logging.getLogger('lsl_socket.py')


class LSL_Socket(Node.Node):
    """
    Closed-Loop System Node to connect to the Lab Streaming Layer (LSL) Interface
    """
    def __init__(self, stream_name, block_size=128, bad_channels=None, store_first_timestamp_to=None, name='lsl_socket'):
        """
        Initialize a LSL socket to receive data send over the Lab Streaming Layer interface

        :param stream_name: Name of the stream to connect to
        :param name: Name of the Node in the Closed-Loop System for debug printing
        """
        super(LSL_Socket, self).__init__(has_inputs=False, name=name)
        self.block_size = block_size
        self.stream_name = stream_name
        self.store_first_timestamp_to = store_first_timestamp_to

        stream = self.find_given_stream()
        stream_inlet = StreamInlet(stream)

        self.mask = np.ones(stream_inlet.channel_count, bool)
        stream_inlet.close_stream()

        if bad_channels is not None and len(bad_channels) > 0:
            self.mask[bad_channels] = False

        self.feeder_process = None
        self.timestamp_stored = False
        logger.info('Connected to stream [{}].'.format(stream_name))

    def find_given_stream(self):
        available_streams = resolve_stream()
        stream_names = list(map(lambda x: x.name(), available_streams))

        stream_index = -1
        try:
            stream_index = stream_names.index(self.stream_name)
        except ValueError:
            logger.error('Stream with name "{}" could not be found. Terminating!'.format(self.stream_name))
            exit(0)

        stream = available_streams[stream_index]
        return stream

    def stream_from_lsl(self):
        """
        Streaming operation which pulls individual samples from LSL
        """
        stream = self.find_given_stream()
        stream_inlet = StreamInlet(stream)

        chunk = np.zeros((32, stream.channel_count()), dtype=np.float32)

        while True:
            _, timestamp = stream_inlet.pull_chunk(max_samples=32, dest_obj=chunk, timeout=FOREVER)

            if not self.timestamp_stored and self.store_first_timestamp_to is not None:
                np.save(self.store_first_timestamp_to, np.array([timestamp[0]]))
                self.timestamp_stored = True

            self.output_data(chunk[:, self.mask])

    def start_processing(self, recurse=True):
        """
        Starts the streaming process.
        """
        logger.info('Start processing')
        if self.feeder_process is None:
            self.feeder_process = multiprocessing.Process(target=self.stream_from_lsl)
            self.feeder_process.start()
        super(LSL_Socket, self).start_processing(recurse)

    def stop_processing(self, recurse=True):
        """
        Stops the streaming process.
        """
        super(LSL_Socket, self).stop_processing(recurse)

        logger.info('Stopping Process [{}]'.format(self.feeder_process.name))
        if self.feeder_process is not None:
            self.feeder_process.terminate()
        self.feeder_process = None
