import sys
sys.path.append('..')
import numpy as np
import argparse
import configparser
import logging
import os
import h5py
import matplotlib
import matplotlib.pyplot as plt
from local.offline import compute_spectrogram, pearson_correlation
from decode import perform_offline_decoding
from train import train
from local.data_loader import Session
from sklearn.model_selection import KFold
from multiprocessing.pool import ThreadPool
from scipy.stats import pearsonr
from scipy.signal import decimate
import pickle
from scipy.io.wavfile import write as wavwrite
from livenodes import Sender, Receiver, GriffinLim
matplotlib.style.use('ggplot')

logger = logging.getLogger('exp1.py')


def train_decode_worker(k, x_train, y_train, x_test, y_test, eeg_sr, audio_sr, bad_channels, norm_factor):
    """
    Worker function for parallel computing
    """
    logger.info('Processing Fold k={}'.format(k))
    _, _, medians, estimators, select = train(x_train, y_train, eeg_sr, audio_sr, bad_channels)

    params = (pickle.dumps(estimators), medians, bad_channels, select)
    reco_spec, out_audio, _, _ = perform_offline_decoding(params, x_test, eeg_sr, norm_factor)

    logger.info('Finished Fold k={}, shape: {}'.format(k, reco_spec.shape))
    return k, reco_spec, y_test, out_audio


class Experiment1:
    """
    Experiment 1 show the general applicability of the proposed method by using a 10-fold cross-validation
    to reconstruct the speech spectrogram of the training data and compare it with chance level.
    """
    def __init__(self, config, session_dir, dest_dir):
        self.session_dir = session_dir
        self.dest_dir = dest_dir
        self.config = config

        # Load training data
        self.sess = Session(session_dir, downsample_audio=False)

    def _construct_datasets_for_run(self, nb_folds=10, randomize=False):
        """
        Perform the 10 fold cross-validation to split the training dataset into disjoint sets. In the process
        of estimating chance level, randomize can be used to break the alignment between neural and audio data.
        """
        kf = KFold(n_splits=nb_folds)
        nb_words = len(self.sess.words)
        np.arange(nb_words)

        # Bad channels should be the same from the training.
        params_file = os.path.join(self.session_dir, 'params.h5')
        with h5py.File(params_file, 'r') as hf:
            bad_channels = hf['bad_channels'][:]

        # Load norm factor
        norm_factor = config.getint('Experiment1', 'griffin_lim_norm')

        arguments = []
        for k, (train, test) in enumerate(kf.split(np.arange(nb_words)), start=1):
            tr_eeg_mask = np.ones(len(self.sess.eeg), dtype=bool)
            tr_audio_mask = np.ones(len(self.sess.audio), dtype=bool)

            va_start = self.sess.word_starts_indices_eeg[test[0]]
            va_stop = self.sess.word_starts_indices_eeg[test[-1]] + 3 * self.sess.eeg_sr
            tr_eeg_mask[va_start:va_stop] = False

            va_start = self.sess.word_starts_indices_audio[test[0]]
            va_stop = self.sess.word_starts_indices_audio[test[-1]] + 3 * self.sess.audio_sr
            tr_audio_mask[va_start:va_stop] = False

            x_train = self.sess.eeg[tr_eeg_mask]
            y_train = self.sess.audio[tr_audio_mask]
            x_test = self.sess.eeg[~tr_eeg_mask]
            y_test = compute_spectrogram(decimate(self.sess.audio[~tr_audio_mask], 3), window_length=0.016)

            x_train = x_train.astype(np.float64)
            minimum = min(len(x_train) / self.sess.eeg_sr, len(y_train) / self.sess.audio_sr)
            x_train = x_train[:int(minimum * self.sess.eeg_sr)]
            y_train = y_train[:int(minimum * self.sess.audio_sr)]

            if randomize:
                r = np.random.randint(0, len(x_train))
                logger.info('Random splitting at index {}'.format(r))
                partition_a = x_train[r:]
                partition_b = x_train[:r]
                x_train = np.vstack([partition_a, partition_b])

            arguments.append((k, x_train, y_train, x_test, y_test, self.sess.eeg_sr,
                              self.sess.audio_sr, bad_channels, norm_factor))
        return arguments

    def _proposed_method_train_decode(self):
        """
        Reconstruct the speech spectrogram using 10 fold cross-validation.
        """
        proposed_method_arguments = self._construct_datasets_for_run()

        with ThreadPool(processes=1) as p:
            results = p.starmap(train_decode_worker, proposed_method_arguments)

        results = sorted(results, key=lambda x: x[0])
        ks, reco, orig, wav = zip(*results)
        reco = np.vstack(reco)
        orig = np.vstack(orig)
        decoded_audio = np.hstack(wav)

        # Store decoded waveforms per trial
        decoding_sr = 16000
        for i, w in enumerate(self.sess.words):
            word_wav = decoded_audio[(i * 3) * decoding_sr:(i * 3+2) * decoding_sr]
            filename = os.path.join(dest_dir, 'reco_wavs', '{:03}-{}.wav'.format(i+1, w))
            wavwrite(filename, decoding_sr, word_wav)

        np.save(os.path.join(self.dest_dir, 'pm_reco.npy'), reco)
        np.save(os.path.join(self.dest_dir, 'orig.npy'), orig)

        pm_dist_means, pm_dist_stds = self._extract_corrs_for_distribution(orig, reco)
        return pm_dist_means, pm_dist_stds

    def _estimate_chance_level(self, nb_runs=100):
        """
        Estimate chance level by using {nb_runs} runs for approximation. For each fold the alignment is
        broken by random splitting and swapping the partitions.
        """
        rc_corrs = []
        for i in range(nb_runs):
            chance_level_arguments = self._construct_datasets_for_run(randomize=True)

            with ThreadPool(processes=1) as p:
                results = p.starmap(train_decode_worker, chance_level_arguments)

            results = sorted(results, key=lambda x: x[0])
            ks, reco, orig, _ = zip(*results)
            reco = np.vstack(reco)
            orig = np.vstack(orig)

            # Store intermediate results
            np.save(os.path.join(self.dest_dir, 'rc_reco_i={:03}.npy').format(i+1), reco)

            _, _, rs = pearson_correlation(orig, reco, return_means=True)
            rc_corrs.append(rs)

        rc_corrs = np.vstack(rc_corrs)
        rc_dist_means = np.mean(rc_corrs, axis=0)
        rc_dist_stds = np.std(rc_corrs, axis=0)

        return rc_dist_means, rc_dist_stds

    def synthesize_specs(self, reco):
        inp = Sender.Sender(reco, 100, 16, asap=True, name='spec_sender')
        gl = GriffinLim.GriffinLimSynthesis(originalFrameSizeMs=16, frameShiftMs=10, sampleRate=16000, melCoeffCount=40,
                                            numReconstructionIterations=8,
                                            normFactor=config.getint('Decoding', 'griffin_lim_norm'))(inp)
        recv = Receiver.Receiver(name='Audio')(gl)

        # Start decoding
        inp.start_processing()
        inp.wait_for_completion()

        # Extract decoded spectrogram
        wav = np.hstack(recv.get_data())

        for i in range(100):
            trial = wav[(i * 3) * 16000:(i * 3 + 2) * 16000]
            word = self.sess.get_trial_by_index(i)[0]
            wav_filename = os.path.join(self.dest_dir, 'resynth', '{:03}-{}.wav'.format(i + 1, word))
            wavwrite(wav_filename, 16000, trial)

    @staticmethod
    def _extract_corrs_for_distribution(orig, reco):
        """
        Extract the distribution of mean correlation coefficients across the frequencies
        """
        n_folds = 5
        kf = KFold(n_splits=5)

        rs = np.zeros((n_folds, orig.shape[1]))
        for k, (train, test) in enumerate(kf.split(orig)):
            o = orig[test, :]
            r = reco[test, :]

            for spec_bin in range(o.shape[1]):
                c = pearsonr(o[:, spec_bin], r[:, spec_bin])[0]
                rs[k, spec_bin] = c

        corr = np.mean(rs, axis=0)
        stds = np.std(rs, axis=0)

        return corr, stds

    def run(self, randomization_runs=100):
        """
        Run the first experiment
        """
        logger.info('Processing computuation for proposed method')
        pm_dist_means, pm_dist_stds = self._proposed_method_train_decode()

        logger.info('Processing computation for estimated chance level.')
        rc_dist_means, rc_dist_stds = self._estimate_chance_level(nb_runs=randomization_runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Execute Experiment 1.')
    parser.add_argument('config', help='Path to experiment config file.')

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
    dest_dir = os.path.join(config['General']['temp_dir'], config['General']['session'], 'exp1')
    nb_randomization_runs = config.getint('Experiment1', 'nb_randomization_runs')

    logger.info('Config file: {}'.format(args.config))
    logger.info('Session: {}'.format(config['General']['session']))
    logger.info('Session dir: {}'.format(session_dir))
    logger.info('Dest dir: {}'.format(dest_dir))
    logger.info('Nb randomization runs: {}'.format(nb_randomization_runs))

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'reco_wavs'), exist_ok=True)

    exp1 = Experiment1(config, session_dir, dest_dir)
    exp1.run(randomization_runs=nb_randomization_runs)
    logger.info('Finished experiment 1.')
