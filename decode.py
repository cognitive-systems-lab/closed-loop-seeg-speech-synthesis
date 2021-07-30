import argparse
import configparser
from livenodes import LDASynthesis, ECogFeatCalc, GriffinLim, lsl_socket, Receiver, ChannelSelector, Sender
from livenodes import Dequantization
import h5py
import logging
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from local.utils import benchmark, extract_sr_from_lsl
from scipy.io.wavfile import write as wavwrite
import os
import platform
from local.utils import in_offline_mode
from local.marker import read_markers
from multiprocessing import Process

logger = logging.getLogger('decode.py')


if platform.system() == 'Linux':
    import jack
    from livenodes import JackAudioSink
elif platform.system() == 'Windows':
    from livenodes import PyAudioSink
else:
    logger.warning('Not supported platform detected. Choose a Windows or Linux operating system. '
                   'System behavior will be undefined and there will be no streaming over loudspeakers.')


@benchmark
def plot_streamed_data(spectrogram, audio, filename):
    ax_spec = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
    ax_audio = plt.subplot2grid((3, 3), (2, 0), colspan=3, rowspan=1)

    m1 = ax_spec.imshow(spectrogram.T, aspect='auto', origin='lower')
    ax_spec.set_title('Decoded speech signal')
    ax_spec.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax_spec.set_ylabel('logMels (dequantized)')
    divider = make_axes_locatable(ax_spec)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(m1, cax=cax, orientation='vertical')

    ax_audio.plot(audio, linewidth=1)
    ax_audio.set_xlim(0, len(audio))
    ax_audio.set_yticks([-(2 ** 15), 0, (2 ** 15) - 1])
    ax_audio.set_yticklabels(['-int16', 0, 'int16'])
    divider = make_axes_locatable(ax_audio)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_visible(False)

    ax_audio.spines['right'].set_position(('outward', 10))
    ax_audio.yaxis.set_ticks_position('right')
    ax_audio.spines['bottom'].set_position(('outward', 10))
    ax_audio.spines['left'].set_color(None)
    ax_audio.spines['top'].set_color(None)
    ax_audio.set_ylabel('Amplitude')
    ax_audio.set_xlabel('Time (in seconds)')

    xticks_in_sec = ax_audio.get_xticks()
    xticks_in_sec = ['{:.02f}'.format(x / 16000.0) for x in xticks_in_sec]
    ax_audio.set_xticklabels(xticks_in_sec)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)


def perform_offline_decoding(params, eeg, sfreq, gl_norm):
    estimators_serialized = params[0]
    medians_array = params[1]
    bad_channels = params[2]
    select = params[3]

    eeg_sender = Sender.Sender(eeg, sfreq, 16, asap=True, name='sEEG-File-Sender')
    logger.info('Using a sampling rate of {} for the sEEG data.'.format(sfreq))
    rec_seeg, rec_spec, rec_audio = setup_decoder(eeg_sender, sfreq, estimators_serialized, medians_array, bad_channels,
                                                  select, gl_norm, include_soundcard=False)

    # Start decoding
    eeg_sender.start_processing()
    eeg_sender.wait_for_completion()

    # Extract decoded spectrogram
    spectrogram = np.array(rec_spec.get_data())

    # Extract streamed Audio
    output_audio = np.hstack(rec_audio.get_data())

    # Also save the streamed sEEG data
    received_sEEG = np.vstack(np.array(rec_seeg.get_data()))

    logger.info('Decoding completed.')
    return spectrogram, output_audio, received_sEEG, sfreq


def perform_online_decoding(config, params, gl_norm):
    estimators_serialized = params[0]
    medians_array = params[1]
    bad_channels = params[2]
    select = params[3]

    run_dir = os.path.join(config['General']['storage_dir'], config['General']['session'],
                           config['Decoding']['run'])

    sfreq = extract_sr_from_lsl(config['Decoding']['stream_name'])
    stream_name = config['Decoding']['stream_name']

    # Store first received timestamp in numpy array
    filename = '.'.join(['first_timestamp', 'npy'])
    filename = os.path.join(run_dir, filename)

    # Use a fixed chunk of 32 samples since the amplifier sends 32 packets with each 32 samples per second
    packet_size = 64 if sfreq == 2048 else 32
    eeg_sender = lsl_socket.LSL_Socket(stream_name, packet_size, store_first_timestamp_to=filename,
                                       name='sEEG_Sender')

    logger.info('Using a sampling rate of {} for the sEEG data.'.format(sfreq))
    rec_seeg, rec_spec, rec_audio = setup_decoder(eeg_sender, sfreq, estimators_serialized, medians_array,
                                                  bad_channels, select, gl_norm, packet_size=packet_size)

    # Start decoding
    eeg_sender.start_processing()

    # Start marker reader only in offline mode
    marker_receiver = Process(target=read_markers, args=(run_dir,))
    marker_receiver.start()
    logger.info('Started Process [{}] with PID [{}] to listen for experiment markers'.format(
        marker_receiver.name, marker_receiver.pid))

    input("Press Enter to stop decoding...\n")
    eeg_sender.stop_processing()
    marker_receiver.terminate()
    marker_receiver.join()
    logger.info('Stopped Process [{}]'.format(marker_receiver.name))

    # Extract decoded spectrogram
    spectrogram = np.array(rec_spec.get_data())

    # Extract streamed Audio
    output_audio = np.hstack(rec_audio.get_data())

    # Also save the streamed sEEG data
    received_sEEG = np.vstack(np.array(rec_seeg.get_data()))

    logger.info('Decoding completed.')
    return spectrogram, output_audio, received_sEEG, sfreq


def setup_decoder(eeg_sender, sfreq, estimators_serialized, medians_array, bad_channels, select, gl_norm=10,
                  packet_size=32, include_soundcard=True, nb_mel_bins=40):
    eeg_select = ChannelSelector.ChannelSelector(exclude=bad_channels, name='BadChannelsExclusion')(eeg_sender)
    eeg_node = ECogFeatCalc.ECogFeatCalc(sfreq, frame_len_ms=50, frame_shift_ms=10,
                                         model_order=4, step_size=5, chunk_size=packet_size)(eeg_select)
    lda_node = LDASynthesis.LDASynthesis(estimators_serialized, select=select)(eeg_node)
    deq_node = Dequantization.Dequantization(medians_array)(lda_node)

    logger.info('Amplifier packet size: {}'.format(packet_size))

    gl_node = GriffinLim.GriffinLimSynthesis(
        originalFrameSizeMs=16, frameShiftMs=10, sampleRate=16000, melCoeffCount=nb_mel_bins,
        numReconstructionIterations=8, normFactor=gl_norm)(deq_node)

    rec_seeg = Receiver.Receiver(name='EEG')(eeg_sender)
    rec_spec = Receiver.Receiver(name='Spectrogram')(deq_node)
    rec_audio = Receiver.Receiver(name='Audio')(gl_node)

    if include_soundcard:  # A soundcard wrapper is only needed in online mode
        if platform.system() == 'Linux':
            try:
                ja_node = JackAudioSink.JackAudioSink(orig_sample_rate=16000,
                                                      allow_fractional_resample=True, block_size=256)(gl_node)
                logger.info('Using JACKAudio to access loudspeakers.')
            except jack.JackError:
                logger.info('Could not start Jack.')

        elif platform.system() == 'Windows':
            pa_node = PyAudioSink.PyAudioSink(orig_sample_rate=16000, block_size=256)(gl_node)
            logger.info('Using PyAudio to access loudspeakers.')

    return rec_seeg, rec_spec, rec_audio


def store_decoding_to_file(spectrogram, output_audio, received_sEEG, sfreq):
    # Save decoding plot to disc
    filename = '.'.join(['decoding', 'png'])
    filename = os.path.join(run_dir, filename)
    plot_streamed_data(spectrogram=spectrogram, audio=output_audio, filename=filename)

    # Save decoded audio
    filename = '.'.join(['audio', 'wav'])
    filename = os.path.join(run_dir, filename)
    wavwrite(filename, 16000, output_audio)
    logger.info('Decoded audio written to {}'.format(filename))

    # Save input sEEG data received from LSL
    filename = '.'.join(['sEEG', 'hdf'])
    filename = os.path.join(run_dir, filename)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('sEEG', data=received_sEEG)
        hf.create_dataset('sEEG_sr', data=sfreq, dtype=np.int32)

    logger.info('Received sEEG written to {}'.format(filename))

    # Save decoded spectrogram
    filename = '.'.join(['spectrogram', 'npy'])
    filename = os.path.join(run_dir, filename)
    np.save(filename, spectrogram)
    logger.info('Decoded spectrogram written to {}'.format(filename))

    # Save used config file
    filename = '.'.join(['decode', 'ini'])
    filename = os.path.join(run_dir, filename)
    with open(filename, 'w') as configfile:
        config.write(configfile)

    logger.info('Decoding configuration written to {}'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Decode a LSL stream of sEEG data using a pretrained linear regression model.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--storage_dir', help='Path to the storage_dir.')
    parser.add_argument('--stream_name', help='Stream name of the sEEG LSL channels.')
    parser.add_argument('--marker_stream_name', help='Stream name of the experiment markers.')
    parser.add_argument('--gl_norm', help='Griffin-Lim norm factor.')
    parser.add_argument('--run', help='Name of the decoding run.')
    parser.add_argument('--session', help='Name of the Session.')
    parser.add_argument('--seeg_file', help='Decode sEEG data read from file instead of from the LSL stream.'
                                            'Useful for output quality comparisons between different trainings.')

    args = parser.parse_args()

    # initialize the config parser
    if not os.path.exists(args.config):
        print('WARNING: File path to the config file is invalid. Please specify a proper path. Script will exit!')
        exit(1)
    config = configparser.ConfigParser()
    config.read(args.config)

    # if optional script arguments change arguments set in config, update them
    if args.storage_dir is not None:
        config['General']['storage_dir'] = args.storage_dir
    if args.stream_name is not None:
        config['Decoding']['stream_name'] = args.stream_name
    if args.marker_stream_name is not None:
        config['Decoding']['marker_stream_name'] = args.marker_stream_name
    if args.gl_norm is not None:
        config['Decoding']['griffin_lim_norm'] = args.gl_norm
    if args.run is not None:
        config['Decoding']['run'] = args.run
    if args.session is not None:
        config['General']['session'] = args.session
    if args.seeg_file is not None:
        config['Development']['seeg_file'] = args.seeg_file

    # check if given session directory exists
    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    if not os.path.isdir(session_dir):
        print('The directory path for the experiment session "{}" seems not to exist. '
              'CHECK THE PATH AGAIN. Script will terminate.'.format(session_dir))
        exit(1)

    # create decoding run directory
    run_dir = os.path.join(config['General']['storage_dir'], config['General']['session'], config['Decoding']['run'])
    try:
        os.makedirs(run_dir, exist_ok=config.getboolean('Decoding', 'overwrite_on_rerun'))
    except FileExistsError:
        print('The directory path "{}" could not be created, since it is already present and the parameter '
              '"overwrite_on_rerun" in the "Training" section is set to False. '
              'Script will exit!'.format(run_dir))
        exit(1)

    # initialize logging handler
    log_file = '.'.join(['decode', 'log'])
    log_file = os.path.join(run_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, 'w+'),
            logging.StreamHandler(sys.stdout)
        ])

    # log script arguments
    params_file = os.path.join(session_dir, 'params.h5')
    if not in_offline_mode(config):
        logger.info('Stream name: {}'.format(config['Decoding']['stream_name']))
    else:
        logger.warning('sEEG file read offline: {}'.format(config['Development']['seeg_file']))
    logger.info('File to load parameters from: {}'.format(params_file))
    logger.info('Session_dir: {}'.format(session_dir))
    logger.info('Run_dir: {}'.format(run_dir))

    # load parameters and the bad channels
    with h5py.File(params_file, 'r') as hf:
        medians_array = hf['medians_array'][:]
        bad_channels = hf['bad_channels'][:]
        p_estimators = hf['estimators'][...].tobytes()
        select = hf['select'][:]

    logger.info('Ignoring channel indices: [' + ' '.join(map(str, bad_channels)) + '].')
    params = (p_estimators, medians_array, bad_channels, select)
    gl_norm = config.getint('Decoding', 'griffin_lim_norm')

    if in_offline_mode(config):
        # Read sEEG data from a file
        with h5py.File(config['Development']['seeg_file'], 'r') as hf:
            eeg = hf['sEEG'][:]
            sfreq = hf['sEEG_sr'][...].reshape((1,))[0]

        # decode from offline data
        spectrogram, output_audio, received_sEEG, sfreq = perform_offline_decoding(
            params=params, eeg=eeg, sfreq=sfreq, gl_norm=gl_norm)
    else:
        # decode from stream
        spectrogram, output_audio, received_sEEG, sfreq = perform_online_decoding(
            config, params=params, gl_norm=gl_norm)
    store_decoding_to_file(spectrogram, output_audio, received_sEEG, sfreq)
