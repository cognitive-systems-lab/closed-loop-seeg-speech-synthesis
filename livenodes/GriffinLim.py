from . import Node
import numpy as np
import scipy.signal as signal
import scipy
import time
import local.MelFilterBank as mel


class GriffinLimSynthesis(Node.Node):
    """
    Takes a continuous stream of logMel spectra as inout and produces an audio stream in form of chunks.
    """
    def __init__(self, originalFrameSizeMs, frameShiftMs, sampleRate, melCoeffCount, numReconstructionIterations = 5, extraContext = 0, cutoff = 7900, normFactor = 1.0, useLogMels = True, name='GriffinLim'):
        """Initializes three ring buffers, one for spectra and two for audio output"""
        super(GriffinLimSynthesis, self).__init__(name=name)
        self.useLogMels = useLogMels

        # Make sure no integer accidents happen
        frameSizeMs = float(originalFrameSizeMs)
        frameShiftMs = float(frameShiftMs)
        sampleRate = float(sampleRate)

        # Frame size and shift
        self.frameShiftMs = frameShiftMs
        self.sampleRate = sampleRate
        self.fftSize = int((frameSizeMs / 1000.0) * self.sampleRate)
        self.frameShift = int((frameShiftMs / 1000.0) * self.sampleRate)

        # Set block length accounting for overlap
        self.contextWidth = int(frameSizeMs / frameShiftMs)
        self.blockLen = self.contextWidth * 2 + 1 + extraContext

        # Length for ring buffers
        self.inputBufferLength = int(self.blockLen * 2.5)
        self.outputBufferLength = int(self.fftSize + self.frameShift * self.blockLen * 2.5)

        # Buffers and positions
        self.inputBuffer = []
        self.outputBuffer = []
        self.windowBuffer = []
        self.inputBufferPos = 0
        self.outputBufferPosMs = 0
        self.framePos = 0
        self.rfc = 0
        self.startTime = time.time()

        # Processing parameters
        self.normFactor = normFactor

        self.fftWindow = scipy.blackman(self.fftSize)
        self.numReconstructionIterations = numReconstructionIterations

        filterOrd = int((sampleRate / 1000.0) * frameShiftMs / 32.0)
        self.filterNumerator, self.filterDenominator = signal.iirfilter(
            filterOrd,
            float(cutoff) / float((sampleRate / 2)),
            btype="lowpass"
        )
        self.filterState = signal.lfiltic(self.filterNumerator, self.filterDenominator, np.array([]))

        specSize = int(int((frameSizeMs / 1000.0) * sampleRate) / 2 + 1)
        self.melFilter = mel.MelFilterBank(specSize, melCoeffCount, sampleRate)

    def stft(self, x):
        """Returns short time fourier transform of a signal x"""
        return np.array([np.fft.rfft(self.fftWindow * x[i:i + self.fftSize]) for i in
                         range(0, len(x) - self.fftSize, self.frameShift)])

    def istft(self, X, lenWaveFile):
        """Returns inverse short time fourier transform of a complex spectrum X"""
        x = scipy.zeros(lenWaveFile)
        for n, i in enumerate(range(0, len(x) - self.fftSize, self.frameShift)):
            x[i:i + self.fftSize] += scipy.real(np.fft.irfft(X[n])) * self.fftWindow
        return x

    def reconstructWavFromSpectrogram(self, spec, lenWaveFile):
        """Returns a reconstructed waveform from the spectrogram
        using the method in
        Griffin, Lim: Signal estimation from modified short-time Fourier transform,
        IEEE Transactions on Acoustics Speech and Signal Processing, 1984

        this version of the class also calculates the inverse mel log
        transform beforehand."""
        if self.useLogMels:
            spec = self.melFilter.fromLogMels(spec)
        else:
            spec = self.melFilter.fromMels(spec)

        # reconstructedWav = np.random.rand(560)
        reconstructedWav = np.random.rand(lenWaveFile)
        for i in range(self.numReconstructionIterations):
            x = self.stft(reconstructedWav)
            z = spec * np.exp(np.angle(x))
            re = self.istft(z, lenWaveFile)
            reconstructedWav[:len(re)] = re
        return reconstructedWav[:len(re)]

    def add_data(self, dataFrame, data_id=0):
        """Add a single frame of data, process it and potentially call callbacks.
           Buffer is allocated on first call."""
        dataFrame = dataFrame.flatten()

        # Allocate buffers, if None
        if self.inputBuffer == []:
            self.inputBuffer = np.zeros((self.inputBufferLength, dataFrame.shape[0]))
            self.outputBuffer = np.zeros(self.outputBufferLength)
            self.windowBuffer = np.zeros(self.outputBufferLength)

        # Add frame to input buffer and increase the framePos which is the write pointer of the ringbuffer
        self.inputBuffer[self.inputBufferPos] = dataFrame
        self.inputBufferPos = (self.inputBufferPos + 1) % self.inputBufferLength
        self.framePos += 1

        # Calculate last index in the output buffer
        previousOutputBufferPos = int((self.outputBufferPosMs / 1000.0) * self.sampleRate)

        # Compute new index in the output buffer
        self.outputBufferPosMs += self.frameShiftMs
        outputBufferPos = int((self.outputBufferPosMs / 1000.0) * self.sampleRate)
        framesShifted = outputBufferPos - previousOutputBufferPos

        # Keep both indices in the range of the ring buffers
        previousOutputBufferPos = previousOutputBufferPos % self.outputBufferLength
        outputBufferPos = outputBufferPos % self.outputBufferLength

        # Force the index of the previous position to be smaller than the current index
        if previousOutputBufferPos >= outputBufferPos:
            previousOutputBufferPos = previousOutputBufferPos - self.outputBufferLength

        # Nothing more do until we have one complete block
        if self.framePos < self.blockLen - self.contextWidth:
            return (np.array([]))

        # Indices for one input block
        bufferIndices = list(range(self.inputBufferPos - self.blockLen + self.contextWidth, self.inputBufferPos))
        bufferIndices = [x + self.inputBufferLength if x < 0 else x for x in bufferIndices]

        # Process block
        reconstructedAudioFrames = self.reconstructWavFromSpectrogram(
            self.inputBuffer[bufferIndices],  # indices of the frames in the block
            self.blockLen * self.frameShift  # blockLen * frameshift in samples
        )

        # Zero out values in the buffer at the current buffer range
        newIndices = list(range(previousOutputBufferPos, outputBufferPos))
        newIndices = [x + self.outputBufferLength if x < 0 else x for x in newIndices]
        self.outputBuffer[newIndices] = np.zeros(len(newIndices))
        self.windowBuffer[newIndices] = np.zeros(len(newIndices))

        # Overlapp-add restored frame
        reconstructedIndices = list(range(outputBufferPos - reconstructedAudioFrames.shape[0], outputBufferPos))
        reconstructedIndices = [x + self.outputBufferLength if x < 0 else x for x in reconstructedIndices]

        overlapWindow = scipy.blackman(reconstructedAudioFrames.shape[0])
        self.outputBuffer[reconstructedIndices] += reconstructedAudioFrames
        self.windowBuffer[reconstructedIndices] += overlapWindow

        # Build finalized frame
        returnIndices = list(range(outputBufferPos - reconstructedAudioFrames.shape[0],
                                   outputBufferPos - reconstructedAudioFrames.shape[0] + framesShifted))
        returnIndices = [x + self.outputBufferLength if x < 0 else x % self.outputBufferLength for x in returnIndices]
        returnBuffer = self.outputBuffer[returnIndices]
        returnWindowBuffer = self.windowBuffer[returnIndices]
        for i in range(len(returnBuffer)):
            if returnWindowBuffer[i] != 0:
                returnBuffer[i] = returnBuffer[i] / returnWindowBuffer[i]

        # Band-pass
        returnBuffer, self.filterState = signal.lfilter(self.filterNumerator, self.filterDenominator, returnBuffer,
                                                        zi=self.filterState)

        # Convert to int16 audio output and return
        self.rfc += len(returnBuffer)
        self.output_data(np.int16(np.clip(returnBuffer / (self.normFactor * 1.01), -0.99, 0.99) * (2 ** 15 - 1)))