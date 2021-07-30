import functools
import time
import sys
import os
from contextlib import contextmanager
import logging
import struct
from pylsl import resolve_stream, StreamInlet
import re
import numpy as np
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


logger = logging.getLogger('utils.py')


def in_offline_mode(config):
    """
    Check if the decode scripts should use the sEEG data from a file instead from the LSL stream
    :param config: config object in which the Development->seeg_file attribute is set
    :return: if set true, otherwise false
    """

    if not config.has_option('Development', 'seeg_file'):
        return False

    if not os.path.exists(config['Development']['seeg_file']):
        print('WARNING: File path to the sEEG file is invalid. Please specify a proper path. Script will exit!')
        exit(1)

    return True


def select_channels(ch_names, good_channels):
    """
    Selects all channels which match at least one specified pattern of the good channels.
    :param ch_names: List of all channel names
    :param good_channels: List of regex patterns to select the good channels
    :return: List of channel names which match the good channel pattern at least once
    """
    patterns = [re.compile(r'^{}$'.format(gc)) for gc in good_channels]

    def matches(ch_name):
        for pattern in patterns:
            if pattern.match(ch_name):
                return True

        return False

    return [ch_name for ch_name in ch_names if matches(ch_name)]


def squeeze_audio_to_float64(audio):
    # Squeeze audio into the range of -1 and 1
    if audio.dtype.kind == 'i':
        # Expecting audio to be in the range -32k / 32k, therefore divide by (2**15)
        audio = audio / (2 ** 15)
        logger.info('Input audio has an integer encoding. Converted to float32.')

    if np.max(audio) > 1:
        logger.warning('Expecting audio to be in the range (-1, 1). '
                       'However, the maximum value is {}'.format(np.max(audio)))

        audio = audio / (2 ** 15)
        logger.info('New maximum after division with {} is {}'.format(2 ** 15, np.max(audio)))

    if np.min(audio) < -1:
        logger.warning('Expecting audio to be in the range (-1, 1). '
                       'However, the minimum value is {}.'.format(np.min(audio)))

        audio = audio / (2 ** 15)
        logger.info('New minimum after division with {} is {}'.format(2 ** 15, np.min(audio)))

    return audio

def check_if_python_shell_is_x64():
    mode = struct.calcsize("P") * 8
    if mode != 64:
        logger.warning('Python Shell is running in x{} and not in x64. '
                       'This might result in MemoryError during data loading.'.format(mode))
    else:
        logger.info('Python Shell running in x{} (as recommended)'.format(mode))


def extract_sr_from_lsl(stream_name):
    stream = resolve_stream('name', stream_name)[0]
    sr = stream.nominal_srate()

    if sr == 0.0:
        logger.warning('Detected an irregular sampling rate. Maybe unspecified in the StreamOutlet.')
    return int(sr)


@contextmanager
def suppress_stdout():
    """ context manager for suppressing printing to stdout for a given block """
    with open(os.devnull, 'w') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout


def benchmark(func):
    """ Print the runtime of the decorated function """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time

        logger.info('Finished method [{}] in {:.4f} seconds.'.format(func.__name__, run_time))
        return value

    return wrapper_timer


def get_warping_path(query_path, reference_path):

    interp_func = interp1d(query_path, reference_path, kind="linear")
    warping_index = interp_func(np.arange(query_path.min(), reference_path.max() + 1)).astype(np.int64)
    warping_index[0] = reference_path.min()

    return warping_index


def dtw_warping(query_spec, reference):
    distance, path = fastdtw(query_spec, reference, dist=euclidean, radius=len(query_spec))
    query, ref = zip(*path)
    query, ref = np.array(query), np.array(ref)
    warping_indices = get_warping_path(query, ref)
    return reference[warping_indices]
