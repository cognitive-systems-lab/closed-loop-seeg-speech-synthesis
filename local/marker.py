import logging
import os
from pylsl import StreamInlet, resolve_stream
import datetime
import sys

logger = logging.getLogger('marker.py')


def read_markers(run_dir):

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ])

    run_filename = '.'.join(['run', 'txt'])
    run_filename = os.path.join(run_dir, run_filename)
    with open(run_filename, 'w') as f:
        f.write('This is just a test.')

    # first resolve a marker stream on the lab network
    logger.info('Looking for a marker stream...')
    streams = resolve_stream('type', 'Markers')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    logger.info('Connected to marker stream [{}]'.format(streams[0].name()))
    marker_filename = '.'.join(['markers', 'csv'])
    marker_filename = os.path.join(run_dir, marker_filename)

    with open(marker_filename, 'w') as f:
        while True:

            sample, timestamp = inlet.pull_sample()
            log_time = datetime.datetime.now().strftime('%d.%M.%y %H:%M:%S')
            f.write('{},{},{}\n'.format(log_time, timestamp, sample[0].strip()))
            f.flush()
