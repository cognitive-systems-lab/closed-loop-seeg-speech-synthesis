import numpy as np
import configparser
from local.data_loader import load_speech_file_by_extension
from local.utils import benchmark, select_channels, squeeze_audio_to_float64
import argparse
import logging
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from local.utils import check_if_python_shell_is_x64
import mne
import h5py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import platform
from local.quantization import compute_borders_logistic, quantize_spectrogram, dequantize_spectrogram
from scipy.stats import spearmanr
import pickle
from local.offline import herff2016_b
from local.offline import compute_spectrogram
from scipy.signal import decimate


logger = logging.getLogger('train.py')


@benchmark
def visualize_train_data(x_train, y_train, filename='train.png', max_number_samples=5000):
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(211)
    m1 = ax.imshow(x_train[0:max_number_samples].T, aspect='auto', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(m1, cax=cax, orientation='vertical')

    ax = fig.add_subplot(212)
    m2 = ax.imshow(y_train[0:max_number_samples].T, aspect='auto', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(m2, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)


@benchmark
def visualize_model_parameters(estimators, filename='coeffs.png'):
    coeffs = np.zeros((len(estimators), len(estimators[0].coef_)))
    for i, est in enumerate(estimators):
        coeffs[i] = est.coef_

    fig = plt.figure(figsize=(5.5, 5))
    ax = fig.add_subplot(111)
    m1 = ax.imshow(coeffs.T, aspect='auto', origin='lower')
    ax.set_title('Visualization of the linear regression coefficients')
    ax.set_xlabel('Linear regression models')
    ax.set_ylabel('Coefficients')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(m1, cax=cax, orientation='vertical')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)


@benchmark
def process_samples_in_network(eeg_sender, aud_sender):
    """
    Computes the ECoG and LogMel features using the node based approach
    """
    eeg_sender.start_processing()
    aud_sender.start_processing()
    eeg_sender.wait_for_completion()
    aud_sender.wait_for_completion()


@benchmark
def quantization(y_train, nb_intervals=8):
    """
    Quantize the logMel spectrogram
    """
    medians, borders = compute_borders_logistic(y_train, nb_intervals=nb_intervals)
    q_spectrogram = quantize_spectrogram(y_train, borders)

    # print if a spec bin does not contain samples for a interval
    for i in range(q_spectrogram.shape[1]):
        diff = np.setdiff1d(np.arange(0, nb_intervals), q_spectrogram[:, i])

        if diff.size > 0:
            logger.info('Spec_bin "{}" misses samples for interval index/indices "{}"'.format(i, str(diff)))

    return medians, borders, q_spectrogram


@benchmark
def feature_selection(x_train, y_train, nb_feats=150):
    """
    Feature selection using correlation
    """
    cs = np.zeros(x_train.shape[1])
    for f in range(x_train.shape[1]):
        if np.isclose(np.sum(x_train[:, f]), 0):
            cs[f] = 0
            continue

        cs[f], p = spearmanr(x_train[:, f], np.mean(y_train, axis=1))
    select = np.argsort(np.abs(cs))[np.max([-nb_feats, -len(cs)]):]
    return select


@benchmark
def train_estimators(estimators, x_train, y_train):
    for mel_bin in range(len(estimators)):
        estimators[mel_bin].fit(x_train, y_train[:, mel_bin])

        if (mel_bin + 1) % 5 == 0:
            logger.info('{:02d} LDAs fitted so far.'.format(mel_bin + 1))

@benchmark
def compute_features(eeg, sfreq_eeg, audio, audio_sr):
    x_train = herff2016_b(eeg, sfreq_eeg, 0.05, 0.01)

    # resample audio to 16kHz
    audio = decimate(audio, 3)
    audio_sr = 16000

    y_train = compute_spectrogram(audio, audio_sr, 0.016, 0.01)
    return x_train, y_train


def train(eeg, audio, sfreq_eeg, sfreq_audio, bad_channels, nb_mel_bins=40):
    # exclude bad channels
    if len(bad_channels) > 0:
        logger.info('EEG original shape: {} x {}'.format(*eeg.shape))
        mask = np.ones(eeg.shape[1], bool)
        mask[bad_channels] = False
        eeg = eeg[:, mask]
        logger.info('EEG truncated shape: {} x {}'.format(*eeg.shape))
    else:
        logger.info('No bad channels specified.')

    x_train, y_train = compute_features(eeg, sfreq_eeg, audio, sfreq_audio)
    y_train = y_train[20:-4]  # Skip 24 samples too align the neural signals to the audio. 20 frames are needed to
                              # first to have all context for one sample. In addition, the window length is 0.05 sec
                              # instead of 0.0016 as with the audio, resulting in 4 more frames. Cutting off in the
                              # beginning aligns the audio to the current frame.

    # Quantize the logMel spectrogram
    medians, borders, q_spectrogram = quantization(y_train, nb_intervals=9)

    # Feature selection using correlation
    select = feature_selection(x_train, y_train)
    x_train = x_train[:, select]

    estimators = [LinearDiscriminantAnalysis() for _ in range(nb_mel_bins)]
    y_train = q_spectrogram

    logger.info('x_train: ' + str(x_train.shape))
    logger.info('y_train: ' + str(y_train.shape))

    # just in case there is still in difference in samples
    minimum = min(len(x_train), len(y_train))
    x_train = x_train[0:minimum, :]
    y_train = y_train[0:minimum, :]
    train_estimators(estimators=estimators, x_train=x_train, y_train=y_train)

    return x_train, y_train, medians, estimators, select


def store_training_to_file(config, x_train, y_train, medians, estimators, bad_channels, select):
    if config.getboolean('Training', 'draw_plots'):
        # visualize train data
        filename = '.'.join(['trainset', 'png'])
        filename = os.path.join(config['General']['storage_dir'], config['General']['session'], filename)
        d_spectrogram = dequantize_spectrogram(y_train, medians)
        visualize_train_data(x_train=x_train, y_train=d_spectrogram, filename=filename)

    # save model parameters to file
    filename = '.'.join(['LDAs', 'pkl'])
    filename = os.path.join(config['General']['storage_dir'], config['General']['session'], filename)
    pickle.dump(estimators, open(filename, 'wb'))

    # Store training features for activation plot
    filename = '.'.join(['training_features', 'npy'])
    filename = os.path.join(config['General']['storage_dir'], config['General']['session'], filename)
    np.save(filename, x_train)

    # store model parameters
    filename = '.'.join(['params', 'h5'])
    filename = os.path.join(config['General']['storage_dir'], config['General']['session'], filename)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('bad_channels', data=bad_channels)
        hf.create_dataset('medians_array', data=medians)
        hf.create_dataset('estimators', data=np.void(pickle.dumps(estimators)))
        hf.create_dataset('select', data=select)

        # Save used config file
        filename = '.'.join(['train', 'ini'])
        filename = os.path.join(config['General']['storage_dir'], config['General']['session'], filename)
        with open(filename, 'w') as configfile:
            config.write(configfile)

    logger.info('Training configuration written to {}'.format(filename))
    logger.info('Training completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train linear regression models on aligned neural and audio data.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--file', help='Comma separated XDF files containing the sEEG data and time aligned audio.')
    parser.add_argument('--session', help='Name of the Session.')
    parser.add_argument('--storage_dir', help='Path to the storage_dir.')
    parser.add_argument('--channels', help='Specify which channels should be used during training and decoding. '
                                           'Accepts a list of regex impressions. The channels will be selected '
                                           'if they match at least one expression. Each regex expression is '
                                           'enclosed in ^EXPRESSION$ to limit its influence.')

    args = parser.parse_args()

    # initialize the config parser
    if not os.path.exists(args.config):
        print('WARNING: File path to the config file is invalid. Please specify a proper path. Script will exit!')
        exit(1)
    config = configparser.ConfigParser()
    config.read(args.config)

    # if optional script arguments change arguments set in config, update them
    if args.file is not None:
        config['Training']['file'] = args.file
    if args.session is not None:
        config['General']['session'] = args.session
    if args.storage_dir is not None:
        config['General']['storage_dir'] = args.storage_dir
    if args.channels is not None:
        config['Training']['channels'] = args.channels

    xdf_files = config['Training']['file'].split(',')

    # create the directory path for storing the session
    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    try:
        os.makedirs(session_dir, exist_ok=config['Training']['overwrite_on_rerun'] == 'True')
    except FileExistsError:
        print('The directory path "{}" could not be created, since it is already present and the parameter '
              '"overwrite_on_rerun" in the "Training" section is set to False. '
              'Script will exit!'.format(session_dir))
        exit(1)

    # initialize logging handler
    log_file = '.'.join(['train', 'log'])
    log_file = os.path.join(config['General']['storage_dir'], config['General']['session'], log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, 'w+'),
            logging.StreamHandler(sys.stdout)
        ])

    # Keep logging clean of pyxdf information
    logging.getLogger('pyxdf.pyxdf').setLevel(logging.WARNING)
    mne.set_log_level("WARNING")

    # log script arguments
    logger.info('XDF files: {}'.format(xdf_files))
    logger.info('Session: {}'.format(config['General']['session']))
    logger.info('Log file: {}'.format(log_file))
    logger.info('Power line noise at {:d} Hz'.format(config.getint('Training', 'power_line')))
    logger.info('Running on {}.'.format(platform.system()))

    # recommended python shell is x64, otherwise a MemoryError can arise during loading
    check_if_python_shell_is_x64()

    # Load the given speech file
    eeg = []
    audio = []

    # EEG and audio sampling rate should not change across multiple runs, same with the channels
    eeg_sr = None
    audio_sr = None
    ch_names = None
    for xdf_file in xdf_files:
        logger.info('Loading {}'.format(xdf_file))
        eeg_i, eeg_sr, audio_i, audio_sr, ch_names = load_speech_file_by_extension(xdf_file)

        # Keep audio between -1 and 1
        audio_i = squeeze_audio_to_float64(audio_i)

        # use float64 for data
        eeg_i = eeg_i.astype(np.float64)

        audio_i = audio_i + np.random.normal(0, 0.0001, len(audio_i))

        # sEEg and audio data might slightly differ in length. Compensate it:
        minimum = min(len(eeg_i) / eeg_sr, len(audio_i) / audio_sr)
        eeg_i = eeg_i[:int(minimum * eeg_sr)]
        audio_i = audio_i[:int(minimum * audio_sr)]

        eeg.append(eeg_i)
        audio.append(audio_i)

        dur = len(eeg_i) / eeg_sr / 60
        logger.info('EEG sampling rate: {}, Audio sampling rate: {}, Amount of speech data: {:.2f} min'.format(eeg_sr,
                                                                                                               audio_sr,
                                                                                                               dur))

    # Merge all recording datasets
    eeg = np.vstack(eeg)
    audio = np.hstack(audio)
    dur = len(eeg) / eeg_sr / 60
    logger.info('In total: {:.2f} min of speech data fro training.'.format(dur))

    # if channels are defined, use only them
    if config['Training']['channels'] is not None:
        regex_patterns = config['Training']['channels'].split(',')
        regex_patterns = list(map(lambda x: x.strip(), regex_patterns))
        sel_channels = select_channels(ch_names, regex_patterns)
    else:
        sel_channels = ch_names

    # mark all non selected channels as bad channels
    bad_channels = [c for c in ch_names if c not in sel_channels]

    ch_types = ['eeg'] * len(ch_names)

    if config.getboolean('Training', 'show_interactive_channel_view'):
        info = mne.create_info(ch_names=ch_names, sfreq=eeg_sr, ch_types=ch_types)
        info['bads'] = bad_channels
        raw = mne.io.RawArray(eeg[0:(60 * eeg_sr)].T, info)

        raw.plot(bad_color='r', title='Select bad channels to exclude', show=True, block=True, scalings='auto')
        bad_channels = raw.info['bads']

    used_channel_names = [c for c in ch_names if c not in bad_channels]
    logger.info('Using the following channels: [' + ' '.join(map(str, used_channel_names)) + '].')

    # Transform list of bad channels to their indices
    bad_channel_indices = [ch_names.index(bc) for bc in bad_channels]

    logger.info('Exclusion of the following bad channel indices: [ ' +
                ' '.join(map(str, bad_channel_indices)) + '].')

    x_train, y_train, medians, estimators, select = train(eeg, audio, eeg_sr, audio_sr, bad_channel_indices)
    store_training_to_file(config, x_train, y_train, medians, estimators, bad_channel_indices, select)
