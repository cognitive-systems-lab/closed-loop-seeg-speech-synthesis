import sys
sys.path.append('..')
import argparse
import configparser
import logging
import os
from local.data_loader import DecodingRun
from local.vad import EnergyBasedVad
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')


logger = logging.getLogger('exp3.py')


class Experiment3:
    """
    Compute amount of speech in decoded acoustic waveforms
    """
    def __init__(self, config, run_dir):
        self.run_dir = run_dir
        self.config = config
        self.vad_frame_context = self.config.getint('Experiment3', 'vad_frames_context')
        self.frame_shift = 0.01

        # Load decoding run
        self.dec_run = DecodingRun(self.run_dir)
        self.vad_mask = None
        self.vad = None

    def _compute_trial_mask(self):
        nb_frame_shift = np.floor(self.frame_shift * self.dec_run.audio_sr).astype(int)
        nb_windows = len(self.dec_run.audio) // nb_frame_shift
        nb_windows -= self.vad_frame_context
        trial_mask = np.zeros(nb_windows, bool)

        word_starts_indices = np.ceil(self.dec_run.word_starts_indices_audio / nb_frame_shift).astype(int)
        word_end_indices = word_starts_indices + (2 * self.dec_run.audio_sr // nb_frame_shift)

        for start, end in zip(word_starts_indices, word_end_indices):
            trial_mask[start:end] = True

        return trial_mask, word_starts_indices[0], word_end_indices[-1]

    def run(self):
        self.vad = EnergyBasedVad(vad_energy_threshold=self.config.getfloat('Experiment3', 'vad_energy_threshold'),
                                  vad_energy_mean_scale=self.config.getint('Experiment3', 'vad_energy_mean_scale'),
                                  vad_frames_context=self.vad_frame_context,
                                  vad_proportion_threshold=self.config.getfloat('Experiment3', 'vad_proportion_threshold'))
        self.vad_mask = self.vad.from_wav(self.dec_run.audio + + np.random.normal(0, 0.0001, len(self.dec_run.audio)),
                                          sampling_rate=self.dec_run.audio_sr)

        trial_mask, experiment_start_index, experiment_end_index = self._compute_trial_mask()

        # Reset everything before the first trial to False since it does not count into the experiment
        self.vad_mask[0:experiment_start_index] = False

        # Reset everything after the last trial to False since it does not count into the experiment
        self.vad_mask[experiment_end_index:] = False

        # Specify the amount in seconds
        amount_of_speech_during_trials = np.count_nonzero(np.logical_and(trial_mask, self.vad_mask)) * self.frame_shift
        amount_of_speech_during_rest = np.count_nonzero(np.logical_and(~trial_mask, self.vad_mask)) * self.frame_shift

        return amount_of_speech_during_trials, amount_of_speech_during_rest


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Execute Experiment 3.')
    parser.add_argument('config', help='Path to config file.')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # initialize logging handler
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)])

    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    dest_dir = os.path.join(config['General']['temp_dir'], config['General']['session'], 'exp3')

    decoding_runs = config['Experiment3']['decoding_runs'].split(',')

    logger.info('Config file: {}'.format(args.config))
    logger.info('Session: {}'.format(config['General']['session']))
    logger.info('Session dir: {}'.format(session_dir))
    logger.info('Decoding runs: {}'.format(decoding_runs))

    os.makedirs(dest_dir, exist_ok=True)

    for run in decoding_runs:
        run_dir = os.path.join(session_dir, run)
        exp3 = Experiment3(config, run_dir)
        amount_of_speech_during_trials, amount_of_speech_during_rest = exp3.run()

        # Save results
        filename = os.path.join(dest_dir, '{}_speech_amount.npy'.format(run))
        results = np.array([amount_of_speech_during_trials, amount_of_speech_during_rest])
        np.save(filename, results)

        # Save VAD
        filename = os.path.join(dest_dir, '{}_run.lab'.format(run))
        exp3.vad.convert_vad_to_lab(filename, exp3.vad_mask)

    logger.info('Finished experiment 3.')
