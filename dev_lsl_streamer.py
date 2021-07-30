from pylsl import StreamInfo, StreamOutlet
from local.data_loader import load_speech_file_by_extension
import time
import signal
import argparse
import configparser
import os
import logging
import random


logger = logging.getLogger('dev_lsl_streamer.py')

# -------------------------------- Constants --------------------------------- #

amp_chunks_size_sr_1024 = 32  # Amplifier sends 32 samples per block (for 2048 sampling rate 64)
amp_chunks_size_sr_2048 = 64
epsilon = 0.0000001  # Spin-waiting sleep time

ifa_words = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
             'groen', 'ook', 'aan', 'schold', 'verlost', 'hij', 'meisjes',
             'een', 'zijn', 'terugvinden', 'bakker', 'redetwisten', 'nachtegalen',
             'braadde', 'nu', 'dichtbij', 'die', 'binnenplaats', 'helft', 'tussen',
             'kin', 'moment', 'op', 'teruggekregen', 'voor', 'vlakbij', 'naar',
             'dat', 'zevenduizend', 'wat', 'donkere', 'de', 'smeekte', 'uittrekken',
             'over', 'zou', 'verdwaald', 'boomstammen', 'tuwiet', 'direct', 'dit',
             'buurt', 'spreuk', 'niet', 'betovering', 'mij', 'je', 'verstijfde',
             'zonlicht', 'al', 'pak', 'werd', 'dan', 'daarna', 'dakker', 'waren',
             'onschuldig', 'het', 'zei', 'onmiddellijk', 'juist', 'was', 'hierop',
             'zandbak', 'wel', 'van', 'doodsbang', 'mooie', 'hoe', 'zanddak', 'haar',
             'helemaal', 'zich', 'bak', 'maar', 'stiekem', 'mooi', 'had', 'tot',
             'vogeltje', 'er', 'vogelkooitje', 'wak', 'te', 'veel', 'maantje', 'alsof',
             'en', 'mijn', 'noordenwind', 'sok', 'vak', 'totdat', 'tak', 'komt',
             'gefluit', 'kwamen', 'bloedrode', 'zo', 'lij', 'sterkste', 'hun', 'om',
             '`s morgens', 'bij', 'dak', 'erheen', 'ze', 'bevrijd', 'uit', 'of',
             'als', 'stilstaan', 'struik', 'hem', 'wegpakte', 'geen', 'vloog', 'in',
             'dauwdruppel', 'wanneer', 'kasteel', 'met', 'sprong', 'door', 'nog', 'deur']

duration_words = 2
duration_cross = 1


# ------------------------- Defining the EEG streamer ------------------------ #

def eeg_streamer(eeg, eeg_sr, stream_markers):

    eeg_info = StreamInfo('dev_sEEG', 'EEG', channel_count=eeg.shape[1], nominal_srate=eeg_sr,
                          channel_format='float32', source_id='amp')
    eeg_outlet = StreamOutlet(eeg_info)

    # sEEG experiment marker stream
    if stream_markers:
        marker_info = StreamInfo('SingleWordsMarkerStream', 'Markers', 1, 0, 'string', 'emuidw22')
        marker_outlet = StreamOutlet(marker_info)

    eeg_sample_index = 0
    time_val = time.time()
    time_val_initial = time_val
    sample_counter = 0
    marker_time = time.time()
    started = False
    word = None
    start_experiment = True
    end_experiment = False
    amp_chunks_size = amp_chunks_size_sr_2048 if eeg_sr == 2048 else amp_chunks_size_sr_1024

    # Stream
    logger.info('Amplifier packet size: {}'.format(amp_chunks_size))
    logger.info('Starting to stream sEEG on channel name [dev_sEEG] with a rate of {}. '
                'Press CTRL-C to terminate...'.format(eeg_sr))

    while True:
        if eeg_sample_index == 0:
            start_experiment = True

        if len(eeg) - amp_chunks_size < eeg_sample_index < len(eeg) - 1:
            end_experiment = True

        chunk = eeg[eeg_sample_index:(eeg_sample_index + amp_chunks_size) % len(eeg)].tolist()

        eeg_sample_index = (eeg_sample_index + amp_chunks_size) % len(eeg)
        eeg_outlet.push_chunk(chunk)

        # Spin-waiting Loop
        while time.time() - time_val < amp_chunks_size / eeg_sr:
            time.sleep(epsilon)

        sample_counter += len(chunk)
        time_val = time_val_initial + sample_counter / eeg_sr

        # Check if markers should be streamed as wel
        if stream_markers:
            if start_experiment:
                marker_outlet.push_sample(['experimentStarted'])
                start_experiment = False
                marker_time = time.time()

            elif end_experiment:
                marker_outlet.push_sample(['experimentEnded'])
                end_experiment = False
                marker_time = time.time()

            else:
                # Push a dummy marker around every 10 seconds
                if (time.time() - marker_time) > duration_words + duration_cross:
                    word = random.choice(ifa_words)
                    marker_outlet.push_sample(['start;' + word])
                    marker_time = time.time()
                    started = True

                # Push a stop dummy marker around 2 seconds after start marker
                if started and (time.time() - marker_time) > duration_words:
                    marker_outlet.push_sample(['end;' + word])
                    started = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract sEEG and Audio from XDF file and stores them into HDF5.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--stream_markers', help='Flag indicating if the markers should be streamed',
                        default=False, action='store_true')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S')

    # initialize the config parser
    if not os.path.exists(args.config):
        logger.error('WARNING: File path to the config file is invalid. Please specify a proper path. Script will exit!')
        exit(1)
    config = configparser.ConfigParser()
    config.read(args.config)

    # Load the given speech file
    eeg, eeg_sr, audio, audio_sr, ch_names = load_speech_file_by_extension(config['Development']['file'])

    if args.stream_markers:
        config['Decoding']['stream_markers'] = True
    stream_markers = False
    if config.has_option('Development', 'stream_markers'):
        stream_markers = config.getboolean('Development', 'stream_markers')

    # Terminate streaming on CTRL-C
    signal.signal(signal.SIGTERM, lambda x, y: exit(0))
    signal.signal(signal.SIGINT, lambda x, y: exit(0))

    eeg_streamer(eeg, eeg_sr, stream_markers=stream_markers)
