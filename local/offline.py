import numpy as np
import scipy
import scipy.signal
import local.MelFilterBank as mel
from local.utils import suppress_stdout
import mne
from scipy.stats import pearsonr
from scipy.signal.windows import hann as hanning
from sklearn.model_selection import KFold


def herff2016_b(eeg, sr, window_length=0.05, window_shift=0.01, line_noise=50, skip_stacking=False):
    """
    Offline computation of the feature extraction paradigma from:
    "Herff, C., et al. Towards direct speech synthesis from ECoG: A pilot study. EMBC 2026"
    This is the version which is compatible with the warm start from the node based system.
    :param eeg: sEEG data in the shape of samples x features
    :param sr: sampling frequency of sEEG data
    :param window_length: Window length
    :param window_shift: Frameshift
    :return: Neural features in the shape of samples x features
    """

    def create_filter(sr, l_freq, h_freq, method='fir', iir_params=None):
        with suppress_stdout():
            iir_params, method = mne.filter._check_method(method, iir_params)
            filt = mne.filter.create_filter(None, sr, l_freq, h_freq, 'auto', 'auto',
                                            'auto', method, iir_params, 'zero', 'hamming', 'firwin')
        return filt

    def extract_high_gamma_50Hz(data, sr, windowLength=0.05, frameshift=0.01):

        # Initialize filters and filter states
        iir_params = {'order': 8, 'ftype': 'butter'}
        hg_filter = create_filter(sr, 70, 170, method='iir', iir_params=iir_params)["sos"]
        fh_filter = create_filter(sr, 102, 98, method='iir', iir_params=iir_params)["sos"]
        sh_filter = create_filter(sr, 152, 148, method='iir', iir_params=iir_params)["sos"]

        hg_state = scipy.signal.sosfilt_zi(hg_filter)
        fh_state = scipy.signal.sosfilt_zi(fh_filter)
        sh_state = scipy.signal.sosfilt_zi(sh_filter)

        hg_state = np.repeat(hg_state, data.shape[1], axis=-1).reshape([hg_state.shape[0], hg_state.shape[1], -1])
        fh_state = np.repeat(fh_state, data.shape[1], axis=-1).reshape([fh_state.shape[0], fh_state.shape[1], -1])
        sh_state = np.repeat(sh_state, data.shape[1], axis=-1).reshape([sh_state.shape[0], sh_state.shape[1], -1])

        warm_start_filling = int(windowLength * sr) - int(frameshift * sr)
        zero_fill = np.zeros([warm_start_filling, data.shape[1]])

        # Initialize high gamma filter state (since online method uses a warm start)
        for i in range(data.shape[1]):
            hg_state[:, :, i] *= data[0, i]

        # Extract high gamma band
        data, hg_state = scipy.signal.sosfilt(hg_filter, data, axis=0, zi=hg_state)

        # Initialize first harmonic filter state (since online method uses a warm start)
        for i in range(data.shape[1]):
            fh_state[:, :, i] *= data[0, i]

        # Update the filter state of the second harmonic on the zero_fill data
        _, sh_state = scipy.signal.sosfilt(sh_filter, zero_fill, axis=0, zi=sh_state)

        # Attenuate first and second harmonic
        data, fh_state = scipy.signal.sosfilt(fh_filter, data, axis=0, zi=fh_state)
        data, sh_state = scipy.signal.sosfilt(sh_filter, data, axis=0, zi=sh_state)
        return data

    def extract_high_gamma_60Hz(data, sr, windowLength=0.05, frameshift=0.01):

        # Initialize filters and filter states
        iir_params = {'order': 8, 'ftype': 'butter'}
        hg_filter = create_filter(sr, 70, 170, method='iir', iir_params=iir_params)["sos"]
        fh_filter = create_filter(sr, 122, 118, method='iir', iir_params=iir_params)["sos"]

        hg_state = scipy.signal.sosfilt_zi(hg_filter)
        sh_state = scipy.signal.sosfilt_zi(fh_filter)

        hg_state = np.repeat(hg_state, data.shape[1], axis=-1).reshape([hg_state.shape[0], hg_state.shape[1], -1])
        sh_state = np.repeat(sh_state, data.shape[1], axis=-1).reshape([sh_state.shape[0], sh_state.shape[1], -1])

        warm_start_filling = int(windowLength * sr) - int(frameshift * sr)
        zero_fill = np.zeros([warm_start_filling, data.shape[1]])

        # Initialize high gamma filter state (since online method uses a warm start)
        for i in range(data.shape[1]):
            hg_state[:, :, i] *= data[0, i]

        # Extract high gamma band
        data, hg_state = scipy.signal.sosfilt(hg_filter, data, axis=0, zi=hg_state)

        # Update the filter state of the second harmonic on the zero_fill data
        _, sh_state = scipy.signal.sosfilt(fh_filter, zero_fill, axis=0, zi=sh_state)

        # Attenuate first and second harmonic
        data, sh_state = scipy.signal.sosfilt(fh_filter, data, axis=0, zi=sh_state)
        return data

    def compute_features(data, windowLength=0.05, frameshift=0.01):
        numWindows = int(np.floor((data.shape[0] - windowLength * sr) / (frameshift * sr))) + 1

        # Compute logarithmic high gamma broadband features
        eeg_features = np.zeros((numWindows, data.shape[1]))
        for win in range(numWindows):
            start_eeg = int(round((win * frameshift) * sr))
            stop_eeg = int(round(start_eeg + windowLength * sr))
            for c in range(data.shape[1]):
                eeg_features[win, c] = np.log(np.sum(data[start_eeg:stop_eeg, c] ** 2) + 0.01)
        return eeg_features

    def stack_features(features, model_order=4, step_size=5):
        eeg_feat_stacked = np.zeros([features.shape[0] - (model_order * step_size), (model_order + 1) * features.shape[1]])
        for f_num, i in enumerate(range(model_order * step_size, features.shape[0])):
            ef = features[i - model_order * step_size:i + 1:step_size, :]
            eeg_feat_stacked[f_num, :] = ef.T.flatten()
        return eeg_feat_stacked

    # Extract HG features and add context information
    if line_noise == 50:
        eeg_feat = extract_high_gamma_50Hz(eeg, sr, windowLength=window_length, frameshift=window_shift)
    else:
        eeg_feat = extract_high_gamma_60Hz(eeg, sr, windowLength=window_length, frameshift=window_shift)

    eeg_feat = compute_features(eeg_feat, windowLength=window_length, frameshift=window_shift)

    if not skip_stacking:
        eeg_feat = stack_features(eeg_feat, model_order=4, step_size=5)
    return eeg_feat


def griffin_lim(spectrogram, win_length=0.05, hop_size=0.01):
    """ Reconstruct an audible acoustic signal using Griffin-Lim approach """

    def griffin_lim_algorithm(spec, lenWaveFile, fftsize=1024, overlap=4.0, numIterations=8):
        """
        Returns a reconstructed waveform from the spectrogram using the method in Griffin, Lim:
            Signal estimation from modified short-time Fourier transform,
            IEEE Transactions on Acoustics Speech and Signal Processing, 1984

        algo described here:
            Bayram, Ilker. "An analytic wavelet transform with a flexible time-frequency covering."
            Signal Processing, IEEE Transactions on 61.5 (2013): 1131-1142.
        """

        def stft(x, fftsize=1024, overlap=4.0):
            """Returns short time fourier transform of a signal x"""
            hop = int(fftsize / overlap)
            w = scipy.hanning(fftsize + 1)[:-1]  # better reconstruction with this trick +1)[:-1]
            return np.array([np.fft.rfft(w * x[i:i + int(fftsize)]) for i in range(0, len(x) - int(fftsize), hop)])

        def istft(X, overlap=4.0):
            """Returns inverse short time fourier transform of a complex spectrum X"""
            fftsize = (X.shape[1] - 1) * 2
            hop = int(fftsize / overlap)
            w = scipy.hanning(fftsize + 1)[:-1]
            x = scipy.zeros(X.shape[0] * hop)
            wsum = scipy.zeros(X.shape[0] * hop)
            for n, i in enumerate(range(0, len(x) - fftsize, hop)):
                x[i:i + fftsize] += scipy.real(np.fft.irfft(X[n])) * w  # overlap-add
                wsum[i:i + fftsize] += w ** 2.0

            return x

        reconstructedWav = np.random.rand(lenWaveFile * 2)
        for i in range(numIterations):
            x = stft(reconstructedWav, fftsize=fftsize, overlap=overlap)
            # print(str(x.shape) + '  ' + str(spec.shape))
            z = spec * np.exp(1j * np.angle(x[:spec.shape[0], :]))  # [:spec.shape[0],:spec.shape[1]]
            re = istft(z, overlap=overlap)
            reconstructedWav[:len(re)] = re
        reconstructedWav = reconstructedWav[:len(re)]
        return reconstructedWav

    audiosr = 16000

    win_len = int(win_length * audiosr)
    overlap = win_length / hop_size

    n_bins = int(win_len / 2 + 1)

    mfb = mel.MelFilterBank(n_bins, spectrogram.shape[1], 16000)  # 401

    # save reconstruction as audio file
    for_reconstruction = mfb.fromLogMels(spectrogram)

    rec_audio = griffin_lim_algorithm(for_reconstruction, for_reconstruction.shape[0] * for_reconstruction.shape[1],
                                      fftsize=win_len,
                                      overlap=overlap)

    scaled = np.int16(rec_audio / np.max(np.abs(rec_audio)) * 32767)

    return scaled


def pearson_correlation(spectrogram_1, spectrogram_2, return_means=False):

    if type(spectrogram_1) is str:
        spectrogram_1 = np.load(spectrogram_1)

    if type(spectrogram_2) is str:
        spectrogram_2 = np.load(spectrogram_2)

    assert spectrogram_1.shape == spectrogram_2.shape, 'Shapes of spectrograms do not match.'

    rs = []
    for spec_bin in range(spectrogram_1.shape[1]):
        r_p, _ = pearsonr(spectrogram_1[:, spec_bin], spectrogram_2[:, spec_bin])
        rs.append(r_p)

    mean = np.mean(rs)
    std = np.std(rs)

    if return_means:
        return mean, std, rs
    else:
        return mean, std


def compute_spectrogram(audio, sr=16000, window_length=0.05, window_shift=0.01, mel_bins=40):
    window_length = int(sr * window_length)
    window_shift = int(sr * window_shift)
    overlap = window_length - window_shift

    # Initialize filling zeros (since online method uses a warm start)
    fill_zeros = np.zeros(overlap)
    audio = np.hstack([fill_zeros, audio])
    num_windows = int(np.floor((len(audio) - overlap) / window_shift))

    segmentation = np.zeros((num_windows, window_length))
    for i in range(num_windows):
        segmentation[i] = audio[i * window_shift: i * window_shift + window_length]

    spectrogram = np.zeros((segmentation.shape[0], window_length // 2 + 1), dtype='complex')
    win = hanning(window_length)
    for i in range(num_windows):
        spectrogram[i] = np.fft.rfft(segmentation[i] * win)

    mfb = mel.MelFilterBank(spectrogram.shape[1], mel_bins, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram


def extract_corrs_for_distribution(orig, reco):
    """
    Extract the distribution of mean correlation coefficients across the frequencies
    """
    n_folds = 10
    kf = KFold(n_splits=n_folds)

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
