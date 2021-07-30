import numpy as np
from scipy.fftpack import dct
import scipy
from local import MelFilterBank as mel


class EnergyBasedVad:
    """
    Energy based VAD computation. Should be equal to compute-vad from Kaldi.

    Arguments:
        log_mels (numpy array): log-Mels which get transformed into MFCCs
        mfcc_coeff (int): Number of MFCC coefficients (exclusive first one)
        vad_energy_threshold (float): If this is set to s, to get the actual threshold we let m be
            the mean log-energy of the file, and use s*m + vad-energy-threshold (float, default = 0.5)
        vad_energy_mean_scale (float): Constant term in energy threshold for MFCC0 for VAD
            (also see --vad-energy-mean-scale) (float, default = 5)
        vad_frames_context (int): Number of frames of context on each side of central frame,
            in window for which energy is monitored (int, default = 0)
        vad_proportion_threshold (int): Parameter controlling the proportion of frames within
            the window that need to have more energy than the threshold (float, default = 0.6)
        export_to_file (str): filename to export the VAD in .lab file format (readable with audacity)
    """
    def __init__(self, vad_energy_threshold=4, vad_energy_mean_scale=1,
                 vad_frames_context=5, vad_proportion_threshold=0.6):

        self.vad_energy_threshold = vad_energy_threshold
        self.vad_energy_mean_scale = vad_energy_mean_scale
        self.vad_frames_context = vad_frames_context
        self.vad_proportion_threshold = vad_proportion_threshold
        self.mfcc_coeff = 13
        self.frame_shift = 0.01
        self.window_length = 0.05

    def from_wav(self, wav, sampling_rate=16000):

        # segment audio into windows
        window_size = int(sampling_rate * self.window_length)
        window_shift = int(sampling_rate * self.frame_shift)
        segmentation = np.arange(0, len(wav) - window_size, window_shift)
        nb_windows = len(segmentation)

        audio_segments = np.zeros((nb_windows, window_size))
        for i, v in enumerate(segmentation):
            audio_segment = wav[v:v + window_size]
            audio_segments[i] = audio_segment

        # create spectrogram from wav
        spectrogram = np.zeros((audio_segments.shape[0], window_size // 2 + 1), dtype='complex')

        win = scipy.hanning(window_size)
        for w in range(audio_segments.shape[0]):
            a = audio_segments[w, :] / (2 ** 15)
            spec = np.fft.rfft(win * a)
            spectrogram[w, :] = spec

        mfb = mel.MelFilterBank(spectrogram.shape[1], 40, sampling_rate)
        log_mels = (mfb.toLogMels(np.abs(spectrogram)))

        return self.from_log_mels(log_mels=log_mels)

    def from_log_mels(self, log_mels):
        self.mfccs = dct(log_mels)
        self.mfccs = self.mfccs[:, 0:self.mfcc_coeff + 2]

        return self.from_mfccs(self.mfccs)

    def from_mfccs(self, mfccs):
        self.mfccs = mfccs
        vad = self._compute_vad()
        return vad

    def _compute_vad(self):
        # VAD computation
        log_energy = self.mfccs[:, 0]
        output_voiced = np.empty(len(self.mfccs), dtype=bool)

        energy_threshold = self.vad_energy_threshold
        if self.vad_energy_mean_scale != 0:
            assert self.vad_energy_mean_scale > 0
            energy_threshold += self.vad_energy_mean_scale * np.sum(log_energy) / len(log_energy)

        assert self.vad_frames_context >= 0
        assert 0.0 < self.vad_proportion_threshold < 1

        for frame_idx in range(0, len(self.mfccs)):
            num_count = 0.0
            den_count = 0.0

            for t2 in range(frame_idx - self.vad_frames_context, frame_idx + self.vad_frames_context):
                if 0 <= t2 < len(self.mfccs):
                    den_count += 1
                    if log_energy[t2] > energy_threshold:
                        num_count += 1

            if num_count >= den_count * self.vad_proportion_threshold:
                output_voiced[frame_idx] = True
            else:
                output_voiced[frame_idx] = False

        return output_voiced

    def convert_vad_to_lab(self, filename, vad):
        last_i = None
        s = None
        r = ''

        for t, i in enumerate(vad):
            if last_i is None:
                last_i = i
                s = 0

            if i != last_i:
                e = t * self.frame_shift  # 10 ms
                r += '{:.2f}\t{:.2f}\t{}\n'.format(s, e, int(last_i))

                s = t * 0.01
                last_i = i

        r += '{:.2f}\t{:.2f}\t{}\n'.format(s, len(vad) * self.frame_shift, int(last_i))

        with open(filename, 'w+') as f:
            f.write(r)
