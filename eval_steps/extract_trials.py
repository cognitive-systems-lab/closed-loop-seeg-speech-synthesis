import sys
sys.path.append('..')
import argparse
import configparser
from local.data_loader import DecodingRun, Session
from scipy.io.wavfile import write as wavwrite
import os
import logging
import pandas as pd


logger = logging.getLogger('extract_trials.py')


def extract_wavs_from_session(session_dir, temp_dir):
    sess = Session(session_dir)

    wavs_dir = os.path.join(temp_dir, 'train_wavs')
    os.makedirs(wavs_dir, exist_ok=True)

    for i, word in enumerate(sess.words):
        audio = sess.get_trial_by_word(word)[2]
        filename = os.path.join(wavs_dir, '{:03}-{}.wav'.format(i + 1, word))
        wavwrite(filename, 16000, audio)


def extract_wavs_from_decoding_trials(run_dir, temp_dir):
    run = DecodingRun(run_dir)
    run_name = os.path.basename(run_dir)

    wavs_dir = os.path.join(temp_dir, '{}_wavs'.format(run_name))
    os.makedirs(wavs_dir, exist_ok=True)

    for i, word in enumerate(run.words):
        audio = run.get_trial_by_word(word)[2]
        filename = os.path.join(wavs_dir, '{:03}-{}.wav'.format(i + 1, word))
        wavwrite(filename, 16000, audio)


def generate_trial_label_file(run_dir, temp_dir):
    run = DecodingRun(run_dir)
    run_name = os.path.basename(run_dir)

    df_dict = {'start': run.trial_starts_in_sec, 'stop': run.trial_starts_in_sec + 2, 'label': run.words}
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(os.path.join(temp_dir, '{}_trials.lab'.format(run_name)), index=False, header=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract decoded trials.')
    parser.add_argument('config', help='Path to experiment config file.')

    args = parser.parse_args()

    # initialize the config parser
    if not os.path.exists(args.config):
        print('WARNING: File path to the config file is invalid. Please specify a proper path. Script will exit!')
        exit(1)
    config = configparser.ConfigParser()
    config.read(args.config)

    # initialize logging handler
    log_file = '.'.join(['train', 'log'])
    log_file = os.path.join(config['General']['storage_dir'], config['General']['session'], log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)])

    logging.getLogger('data_loader.py').setLevel(logging.WARNING)

    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    decoding_runs = [os.path.join(session_dir, run_dir) for run_dir in os.listdir(session_dir)
                     if os.path.isdir(os.path.join(session_dir, run_dir))]
    temp_dir = os.path.join(config['General']['temp_dir'], config['General']['session'])
    os.makedirs(temp_dir, exist_ok=True)

    logger.info('Processing training data'.format(os.path.basename(config['General']['session'])))

    # Extract wavs from training session
    extract_wavs_from_session(session_dir=session_dir, temp_dir=temp_dir)

    # Extract trials from decoding runs
    for run_dir in decoding_runs:
        try:
            logger.info('Processing wavs of {}'.format(os.path.basename(run_dir)))

            # Extract wavs
            extract_wavs_from_decoding_trials(run_dir=run_dir, temp_dir=temp_dir)

            # Generate .lab file
            generate_trial_label_file(run_dir=run_dir, temp_dir=temp_dir)

        except Exception as e:
            logger.warning('Skipping {} due to a caused exception: {}'.format(run_dir, str(e)))
