from local import xdf
import bisect
from local.utils import benchmark
import h5py
import numpy as np
import os
import logging
import pandas as pd
from scipy.signal import decimate
from scipy.io.wavfile import read as wavread

logger = logging.getLogger('data_loader.py')


@benchmark
def load_hdf5(hdf_file, return_markers=True):
    with h5py.File(hdf_file, 'r') as hf:
        ecog = hf['sEEG'][:]
        audio = hf['Audio'][:].astype(np.float64)
        ecog_sr = hf['sEEG_sr'][...].reshape((1,))[0]
        audio_sr = hf['Audio_sr'][...].reshape((1,))[0]

        try:
            ch_names = hf['ch_names'][:].tolist()
            ch_names = list(map(lambda x: x.decode('utf-8'), ch_names))
        except KeyError:
            logger.info('No channels names found in {}. Falling back to channel indices as names.'.format(hdf_file))
            ch_names = ['ch_{:03d}'.format(i) for i in range(ecog.shape[1])]

        if return_markers:
            markers = hf['markers'][:].tolist()
            markers = [[marker[0].decode('utf-8')] for marker in markers]
            return ecog, ecog_sr, audio, audio_sr, ch_names, markers
        else:
            return ecog, ecog_sr, audio, audio_sr, ch_names


@benchmark
def load_XDF(xdf_file, return_markers=False):
    # Load XDF file
    streams = xdf.load_xdf(xdf_file)  # , verbose=False)
    streamToPosMapping = {}

    marker_stream_name = 'SingleWordsMarkerStream'
    for pos in range(0, len(streams[0])):
        stream = streams[0][pos]['info']['name']
        streamToPosMapping[stream[0]] = pos

        if streams[0][pos]['info']['type'][0] == 'Markers':
            marker_stream_name = stream[0]

    logger.info('Marker stream name: {}'.format(marker_stream_name))

    # Get sEEG
    eeg = streams[0][streamToPosMapping['Micromed']]['time_series']
    offset = float(streams[0][streamToPosMapping['Micromed']]['info']['created_at'][0])
    eeg_ts = streams[0][streamToPosMapping['Micromed']]['time_stamps'].astype('float')  # +offset
    eeg_sr = int(streams[0][streamToPosMapping['Micromed']]['info']['nominal_srate'][0])

    # Get channel info
    chNames = []
    for ch in streams[0][streamToPosMapping['Micromed']]['info']['desc'][0]['channels'][0]['channel']:
        chNames.append(ch['label'])

    # Load Audio
    audio = streams[0][streamToPosMapping['AudioCaptureWin']]['time_series']
    offset_audio = float(streams[0][streamToPosMapping['AudioCaptureWin']]['info']['created_at'][0])
    audio_ts = streams[0][streamToPosMapping['AudioCaptureWin']]['time_stamps'].astype('float')  # +offset
    audio_sr = int(streams[0][streamToPosMapping['AudioCaptureWin']]['info']['nominal_srate'][0])

    # Load Marker stream
    markers = streams[0][streamToPosMapping[marker_stream_name]]['time_series']
    offset_marker = float(streams[0][streamToPosMapping[marker_stream_name]]['info']['created_at'][0])
    marker_ts = streams[0][streamToPosMapping[marker_stream_name]]['time_stamps'].astype('float')  # -offset


    def locate_pos(available_freqs, target_freq):
        pos = bisect.bisect_right(available_freqs, target_freq)
        if pos == 0:
            return 0
        if pos == len(available_freqs):
            return len(available_freqs) - 1
        if abs(available_freqs[pos] - target_freq) < abs(available_freqs[pos - 1] - target_freq):
            return pos
        else:
            return pos - 1

    # Get Experiment time
    i = 0
    while markers[i][0] != 'experimentStarted':
        i += 1
    eeg_start = locate_pos(eeg_ts, marker_ts[i])
    audio_start = locate_pos(audio_ts, eeg_ts[eeg_start])
    while markers[i][0] != 'experimentEnded':
        i += 1
    eeg_end = locate_pos(eeg_ts, marker_ts[i])
    audio_end = locate_pos(audio_ts, eeg_ts[eeg_end])
    markers = markers[:i]
    marker_ts = marker_ts[:i]

    eeg = eeg[eeg_start:eeg_end, :]
    eeg_ts = eeg_ts[eeg_start:eeg_end]
    audio = audio[audio_start:audio_end, :].astype(np.float64)
    audio_ts = audio_ts[audio_start:audio_end]

    ch_names = [cn[0] for cn in chNames]
    if return_markers:
        return eeg, eeg_sr, audio[:, 0], audio_sr, ch_names, markers
    else:
        return eeg, eeg_sr, audio[:, 0], audio_sr, ch_names


def load_only_eeg_from_other_tasks(xdf_file):
    # Load XDF file
    streams = xdf.load_xdf(xdf_file)  # , verbose=False)
    streamToPosMapping = {}

    marker_stream_name = 'SingleWordsMarkerStream'
    for pos in range(0, len(streams[0])):
        stream = streams[0][pos]['info']['name']
        streamToPosMapping[stream[0]] = pos

        if streams[0][pos]['info']['type'][0] == 'Markers':
            marker_stream_name = stream[0]

    logger.info('Marker stream name: {}'.format(marker_stream_name))

    # Get sEEG
    eeg = streams[0][streamToPosMapping['Micromed']]['time_series']
    offset = float(streams[0][streamToPosMapping['Micromed']]['info']['created_at'][0])
    eeg_ts = streams[0][streamToPosMapping['Micromed']]['time_stamps'].astype('float')  # +offset
    eeg_sr = int(streams[0][streamToPosMapping['Micromed']]['info']['nominal_srate'][0])

    # Get channel info
    chNames = []
    for ch in streams[0][streamToPosMapping['Micromed']]['info']['desc'][0]['channels'][0]['channel']:
        chNames.append(ch['label'])

    # Load Marker stream
    markers = streams[0][streamToPosMapping[marker_stream_name]]['time_series']
    offset_marker = float(streams[0][streamToPosMapping[marker_stream_name]]['info']['created_at'][0])
    marker_ts = streams[0][streamToPosMapping[marker_stream_name]]['time_stamps'].astype('float')  # -offset


    def locate_pos(available_freqs, target_freq):
        pos = bisect.bisect_right(available_freqs, target_freq)
        if pos == 0:
            return 0
        if pos == len(available_freqs):
            return len(available_freqs) - 1
        if abs(available_freqs[pos] - target_freq) < abs(available_freqs[pos - 1] - target_freq):
            return pos
        else:
            return pos - 1

    # Get Experiment time
    i = 0
    while markers[i][0] != 'experimentStarted':
        i += 1
    eeg_start = locate_pos(eeg_ts, marker_ts[i])
    while markers[i][0] != 'experimentEnded':
        i += 1
    eeg_end = locate_pos(eeg_ts, marker_ts[i])

    markers = markers[:i]
    marker_ts = marker_ts[:i]

    eeg = eeg[eeg_start:eeg_end, :]
    eeg_ts = eeg_ts[eeg_start:eeg_end]

    ch_names = [cn[0] for cn in chNames]
    return eeg, eeg_sr, ch_names


def load_speech_file_by_extension(speech_file):
    """
    Load the given speech file based on their extension [.xdf, .h5, .hdf, .hdf5].
    :param speech_file: Path to the speech file
    :return: eeg, eeg_sr, audio, audio_sr, ch_names
    """
    extension = os.path.splitext(speech_file)[1][1:]
    eeg, eeg_sr, audio, audio_sr, ch_names = [None] * 5
    if extension in ['h5', 'hdf', 'hdf5']:
        logger.info('File format is HDF5.')
        eeg, eeg_sr, audio, audio_sr, ch_names = load_hdf5(speech_file)
    elif extension == 'xdf':
        logger.info('File format is XDF.')
        eeg, eeg_sr, audio, audio_sr, ch_names = load_XDF(speech_file)
    else:
        logger.info('Unknown file format [{}]'.format(extension))
        exit(1)

    return eeg, eeg_sr, audio, audio_sr, ch_names


class Session:

    def __init__(self, session_dir, complete_trial_duration=3, downsample_audio=True):
        self.session_dir = session_dir

        # Load all data files
        filename = os.path.join(self.session_dir, 'speech1.hdf')
        self.eeg, self.eeg_sr, audio, self.audio_sr, self.ch_names, self.markers = load_hdf5(filename, return_markers=True)
        if downsample_audio:
            audio = decimate(audio, 3)
            self.audio_sr = 16000
        self.audio = audio + np.random.normal(0, 0.0001, len(audio))

        self.words = [m[0][6:].strip() for m in self.markers if m[0].startswith('start;')]
        if len(self.words) != 100:
            logger.warning('Number of words does not match 100.')

        self.word_starts_indices_eeg = [(t * complete_trial_duration) * self.eeg_sr for t in range(len(self.words))]
        self.word_starts_indices_audio = [(t * complete_trial_duration) * self.audio_sr for t in range(len(self.words))]

    def _trial_generator(self, duration=2):

        for i in range(len(self.words)):
            trial_start_eeg = self.word_starts_indices_eeg[i]
            trial_start_audio = self.word_starts_indices_audio[i]

            trial_end_eeg = trial_start_eeg + (duration * self.eeg_sr)
            trial_end_audio = trial_start_audio + (duration * self.audio_sr)

            print(trial_start_audio, trial_end_audio)

            w = self.words[i]
            eeg_segment = self.eeg[trial_start_eeg:trial_end_eeg]
            audio_segment = self.audio[trial_start_audio:trial_end_audio]

            yield w, eeg_segment, audio_segment

    def get_trial_generator(self, duration=2):
        return self._trial_generator(duration=duration)

    def get_trial_by_index(self, index, include_rest=False):
        duration = 3 if include_rest else 2
        word_start_index_eeg = self.word_starts_indices_eeg[index]
        word_end_index_eeg = word_start_index_eeg + (duration * self.eeg_sr)

        word_start_index_audio = self.word_starts_indices_audio[index]
        word_end_index_audio = word_start_index_audio + (duration * self.audio_sr)
        word = self.words[index]

        return word, self.eeg[word_start_index_eeg:word_end_index_eeg], \
               self.audio[word_start_index_audio:word_end_index_audio]

    def get_trial_by_word(self, word, include_rest=False):
        index = self.words.index(word)
        return self.get_trial_by_index(index, include_rest=include_rest)


class DecodingRun:

    def __init__(self, run_dir):
        self.run_dir = run_dir

        # Find decoded wav file
        wav_filename = '.'.join(['audio', 'wav'])
        wav_filename = os.path.join(self.run_dir, wav_filename)

        # Load decoded audio
        self.audio_sr, self.audio = wavread(wav_filename)
        logger.info('Decoded audio loaded')

        # Load first timestamp
        ft_filename = '.'.join(['first_timestamp', 'npy'])
        ft_filename = os.path.join(self.run_dir, ft_filename)
        first_timestamp = np.load(ft_filename)

        # Load markers csv
        mk_filename = '.'.join(['markers', 'csv'])
        mk_filename = os.path.join(self.run_dir, mk_filename)
        markers = pd.read_csv(mk_filename, header=None, names=['RecordedAt', 'MonotonicTime', 'Label'])
        markers['Seconds'] = markers.MonotonicTime - first_timestamp

        round_seconds = list(map(lambda x: round(x, 2), markers[markers.Label.str.startswith('start;')].Seconds.values))
        round_seconds = np.array(round_seconds)
        self.trial_starts_in_sec = round_seconds
        self.word_starts_indices_audio = (round_seconds * self.audio_sr).astype(int)
        self.words = markers[markers.Label.str.startswith('start;')].Label.apply(lambda x: x[6:]).values.tolist()

        # Load sEEG
        filename = os.path.join(self.run_dir, 'sEEG.hdf')
        with h5py.File(filename, 'r') as f:
            self.eeg = f['sEEG'][...]
            self.eeg_sr = int(f['sEEG_sr'][...])

        self.word_starts_indices_eeg = (round_seconds * self.eeg_sr).astype(int)

    def _trial_generator(self, duration=2):

        for i in range(len(self.words)):
            trial_start_eeg = self.word_starts_indices_eeg[i]
            trial_start_audio = self.word_starts_indices_audio[i]

            trial_end_eeg = trial_start_eeg + (duration * self.eeg_sr)
            trial_end_audio = trial_start_audio + (duration * self.audio_sr)

            print(trial_start_audio, trial_end_audio)

            w = self.words[i]
            eeg_segment = self.eeg[trial_start_eeg:trial_end_eeg]
            audio_segment = self.audio[trial_start_audio:trial_end_audio]

            yield w, eeg_segment, audio_segment

    def get_trial_generator(self, duration=2):
        return self._trial_generator(duration=duration)

    def get_trial_by_index(self, index, include_rest=False):
        duration = 3 if include_rest else 2
        word_start_index_eeg = self.word_starts_indices_eeg[index]
        word_end_index_eeg = word_start_index_eeg + (duration * self.eeg_sr)

        word_start_index_audio = self.word_starts_indices_audio[index]
        word_end_index_audio = word_start_index_audio + (duration * self.audio_sr)
        word = self.words[index]

        return word, self.eeg[word_start_index_eeg:word_end_index_eeg], \
               self.audio[word_start_index_audio:word_end_index_audio]

    def get_trial_by_word(self, word, include_rest=False):
        index = self.words.index(word)
        return self.get_trial_by_index(index, include_rest=include_rest)
