import sys
sys.path.append('..')
import argparse
import configparser
import logging
import os
import numpy as np
import time
import sys
from local.offline import compute_spectrogram, pearson_correlation
from multiprocessing.pool import ThreadPool
from decode import perform_offline_decoding
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from local.data_loader import load_XDF, load_only_eeg_from_other_tasks, Session, DecodingRun
from scipy.interpolate import interp1d
import h5py
import pickle


logger = logging.getLogger('exp2')


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


def chance_level_worker(config, run, r, seeg_data, orig):
    logger.info('Processing run {} (split occurred at {})'.format(run, r))
    t0 = time.time()

    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    params_filename = os.path.join(session_dir, 'params.h5')
    norm_factor = config.getint('Experiment2', 'griffin_lim_norm')

    with h5py.File(params_filename, 'r') as hf:
        medians = hf['medians_array'][:]
        bad_channels = hf['bad_channels'][:]
        estimators = hf['estimators'][...].tobytes()
        select = hf['select'][:]
    params = (estimators, medians, bad_channels, select)

    reco_spec, _, _, _ = perform_offline_decoding(params=params, eeg=seeg_data, sfreq=2048, gl_norm=norm_factor)

    warped_orig = dtw_warping(reco_spec, orig)
    corr = pearson_correlation(warped_orig, reco_spec)[0]

    logger.info('Finished run {} after {:02f} seconds (score: {})'.format(run, time.time() - t0, corr))
    return run, corr


class Experiment_2:

    def __init__(self, config, session_dir, run_dir, other_tasks, dest_dir):
        self.session_dir = session_dir
        self.dest_dir = dest_dir
        self.run_dir = run_dir
        self.config = config
        self.audio_sr = 16000

        # Load decoding run
        self.dec_run = DecodingRun(self.run_dir)

        # Load training data
        self.sess = Session(session_dir)

        # Load other tasks sEEG data
        other_tasks_eeg = []
        for ot in other_tasks:
            ot_data = load_only_eeg_from_other_tasks(os.path.join(session_dir, ot))[0]

            logger.info('{} loaded'.format(ot))
            other_tasks_eeg.append(ot_data)

        self.other_tasks_eeg = np.vstack(other_tasks_eeg)

    def _estimate_chance_level(self, runs=100):
        training_words = list(zip(self.sess.word_starts_indices_audio, self.sess.words))

        arguments = []  # Stores for each run all the information which is necessary to perform decoding on
                        # a random segment of the sEEG data
        for i in range(runs):
            cutoff = np.random.randint(0, len(self.other_tasks_eeg) - 2 * self.dec_run.eeg_sr)
            # cutoff 2 seconds of sEEG data
            seeg_data = self.other_tasks_eeg[cutoff:cutoff + 2 * self.dec_run.eeg_sr, :]

            # Extract 2 seconds of speech data of the reference word
            word_starts_index_audio, orig_word = training_words[i % len(training_words)]
            word_end_index_audio = word_starts_index_audio + 2 * self.sess.audio_sr
            orig = self.sess.audio[word_starts_index_audio:word_end_index_audio]
            orig = compute_spectrogram(orig, self.sess.audio_sr, window_length=0.016)

            arguments.append((config, i + 1, cutoff, seeg_data, orig))

        # Compute DTW correlation results
        with ThreadPool(processes=1) as p:
            results = p.starmap(chance_level_worker, arguments)

        _, corrs = zip(*results)
        return np.array(corrs)

    def _compute_correlation_of_matching_trials(self):
        words_in_training_phase = set(self.sess.words)
        words_in_decoding_phase = set(self.dec_run.words)

        intersecting_words = words_in_training_phase & words_in_decoding_phase
        corrs_matching_pairs = []
        for word in intersecting_words:
            logger.info(word)
            train_trial_audio = self.sess.get_trial_by_word(word)[2]
            decode_trial_audio = self.dec_run.get_trial_by_word(word)[2]

            train_trial_logMels = compute_spectrogram(train_trial_audio, self.sess.audio_sr, window_length=0.016)
            decode_trial_logMels = compute_spectrogram((decode_trial_audio / (2 ** 15)).astype(float), self.dec_run.audio_sr, window_length=0.016)

            warped_ref = dtw_warping(decode_trial_logMels, train_trial_logMels)

            corr = pearson_correlation(warped_ref, decode_trial_logMels)[0]
            corrs_matching_pairs.append(corr)

        return corrs_matching_pairs

    def run(self, runs=100, which='both'):
        run = os.path.basename(self.run_dir)

        if which in ['both', 'chance_only']:
            logger.info('Estimating chance level on other tasks')
            chance_level_estimation = self._estimate_chance_level(runs=runs)
            nb_nan = np.count_nonzero(np.isnan(chance_level_estimation))
            if nb_nan > 0:
                logger.warning('{} estimation runs returned NaN values. These entries will be skipped.'.format(nb_nan))
                chance_level_estimation = chance_level_estimation[~np.isnan(chance_level_estimation)]

            filename = os.path.join(self.dest_dir, 'exp2_{}_chance.npy'.format(run))
            np.save(filename, chance_level_estimation)

        if which in ['both', 'pm_only']:
            logger.info('Estimate pearson correlations on matching pairs')
            decoding = self._compute_correlation_of_matching_trials()
            logger.info('{} words are present in decoding run and training session.'.format(len(decoding)))

            filename = os.path.join(self.dest_dir, 'exp2_{}_pm.npy'.format(run))
            np.save(filename, decoding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Execute Experiment 2.')
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

    logging.getLogger('decoder.py').setLevel(logging.WARNING)
    logging.getLogger('ECoGFeatCalc.py').setLevel(logging.WARNING)

    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    dest_dir = os.path.join(config['General']['temp_dir'], config['General']['session'], 'exp2')

    decoding_runs = config['Experiment2']['decoding_runs'].split(',')
    other_xdf_files = config['Experiment2']['other_xdf'].split(',')
    nb_randomization_runs = config.getint('Experiment2', 'nb_randomization_runs')

    logger.info('Config file: {}'.format(args.config))
    logger.info('Session: {}'.format(session_dir))
    logger.info('Decoding run(s): {}'.format(decoding_runs))
    logger.info('Nb randomization runs: {}'.format(nb_randomization_runs))
    logger.info('(Other) Baseline tasks: {}'.format(other_xdf_files))
    logger.info('Which evaluation: {}'.format(config['Experiment2']['which']))

    os.makedirs(dest_dir, exist_ok=True)

    for decoding_run in decoding_runs:
        run_dir = os.path.join(session_dir, decoding_run)
        logger.info('Processing decoding run "{}".'.format(decoding_run))

        exp2 = Experiment_2(config, session_dir, run_dir, other_tasks=other_xdf_files, dest_dir=dest_dir)
        exp2.run(runs=nb_randomization_runs, which=config['Experiment2']['which'])

    logger.info('Finished experiment 2.')
