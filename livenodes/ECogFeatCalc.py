import numpy as np
import mne.filter
from . import Node
from . import FrameBuffer
from . import LambdaNode
import os
import sys
from contextlib import contextmanager
import logging


logger = logging.getLogger('ECoGFeatCalc.py')


class ECogFeatCalc(Node.Node):
    """
    Takes a continuous stream of ECoG-data as input and outputs ECoG features.
    """
    def __init__(self, sample_rate, frame_len_ms, frame_shift_ms, model_order=4, step_size=5,
                 line_noise=50, warm_start=True, chunk_size=32, has_inputs=True, name='ECogFeatCalc'):
        """
        Initializes all the nodes used to run-on calculate ECog features.
        """
        super().__init__(name=name, has_inputs=has_inputs)

        self.sample_rate = sample_rate
        self.frame_len_ms = frame_len_ms
        self.frame_shift_ms = frame_shift_ms
        self.model_order = model_order
        self.step_size = step_size
        self.iir_params = None

        logger.info('Framelength in ms: ' + str(self.frame_len_ms))
        logger.info('Frameshift in ms: ' + str(self.frame_shift_ms))
        logger.info('Samplerate: ' + str(self.sample_rate))

        self.first_harmonic_filter = None
        self.second_harmonic_filter = None

        with self.suppress_stdout():
            # Extract High-Gamma Band
            self.high_gamma_filter = self._create_filter(self.sample_rate, 70, 170, method='iir',
                                                         iir_params={'order': 8, 'ftype': 'butter'})["sos"]

            # Setup filters for EU zone
            if line_noise == 50:
                # Remove first harmonic of the line noise
                self.first_harmonic_filter = self._create_filter(self.sample_rate, 102, 98, method='iir',
                                                                 iir_params={'order': 8, 'ftype': 'butter'})["sos"]

                # Remove second harmonic of the line noise (only available in EU)
                self.second_harmonic_filter = self._create_filter(self.sample_rate, 152, 148, method='iir',
                                                                  iir_params={'order': 8, 'ftype': 'butter'})["sos"]

            # Setup filters for US zone
            if line_noise == 60:
                # Remove first harmonic of the line noise
                self.first_harmonic_filter = self._create_filter(self.sample_rate, 122, 118, method='iir',
                                                                 iir_params={'order': 8, 'ftype': 'butter'})["sos"]

        self.stack_buffer_size_ms = self.model_order * self.step_size + 1
        self.first_frame = True

        # Set up sub-graph
        one_frame = (1.0 / self.sample_rate) * 1000.0
        one_frame *= chunk_size
        self.fb_high_gamma = FrameBuffer.FrameBuffer(one_frame, one_frame, self.sample_rate,
                                                     filter_coefficients=self.high_gamma_filter, name=name+'.HighGamma')

        # Connect for EU
        if line_noise == 50:
            one_frame = (1.0 / self.sample_rate) * 1000.0
            one_frame *= chunk_size
            self.fb_first_harmonic = FrameBuffer.FrameBuffer(one_frame, one_frame, self.sample_rate,
                                                             filter_coefficients=self.first_harmonic_filter,
                                                             name=name + '.FirstHarmonic')(self.fb_high_gamma)

            self.fb_second_harmonic = FrameBuffer.FrameBuffer(self.frame_len_ms, self.frame_shift_ms,
                                                              self.sample_rate,
                                                              filter_coefficients=self.second_harmonic_filter,
                                                              warm_start=warm_start,
                                                              name=name + '.SecondHarmonic')(self.fb_first_harmonic)

            self.ecog_feat_calc = LambdaNode.LambdaNode(self.frame_extract_hg,
                                                        name=name+'.HGExtraction')(self.fb_second_harmonic)

        # Connect for US
        if line_noise == 60:
            one_frame = (1.0 / self.sample_rate) * 1000.0
            one_frame *= chunk_size
            self.fb_first_harmonic = FrameBuffer.FrameBuffer(self.frame_len_ms, self.frame_shift_ms, self.sample_rate,
                                                             filter_coefficients=self.first_harmonic_filter,
                                                             warm_start=warm_start,
                                                             name=name + '.FirstHarmonic')(self.fb_high_gamma)

            self.ecog_feat_calc = LambdaNode.LambdaNode(self.frame_extract_hg,
                                                        name=name+'.HGExtraction')(self.fb_first_harmonic)

        self.stack_buff = FrameBuffer.FrameBuffer(self.stack_buffer_size_ms, 1, 1000, warm_start=warm_start,
                                                  name=name+'.StackBuffer')(self.ecog_feat_calc)
        self.stacker = LambdaNode.LambdaNode(self.stack_features, name=name + '.Stacker')(self.stack_buff)

        # Set up subgraph passthrough.
        self.set_passthrough(self.fb_high_gamma, self.stacker)

    @staticmethod
    @contextmanager
    def suppress_stdout():
        with open(os.devnull, 'w') as devnull:
            stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = stdout

    @staticmethod
    def frame_extract_hg(data):
        """
        Extracts high gamma features
        """

        ecog_feat = np.log(np.sum(data**2, axis=0)+0.01)
        return np.array([ecog_feat.T])

    def _create_filter(self, sfreq, l_freq, h_freq, filter_length='auto',
                       l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                       method='fir', iir_params=None, phase='zero',
                       fir_window='hamming', fir_design='firwin'):

        self.iir_params, method = mne.filter._check_method(method, iir_params)
        filt = mne.filter.create_filter(None, sfreq, l_freq, h_freq, filter_length, l_trans_bandwidth,
                                        h_trans_bandwidth, method, self.iir_params, phase, fir_window, fir_design)

        return filt

    def stack_features(self, data):
        """
        Skips features for stacking
        """

        data = data[::self.step_size]
        data = data.T.flatten()
        return data
