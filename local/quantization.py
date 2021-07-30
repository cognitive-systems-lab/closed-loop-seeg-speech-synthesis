import numpy as np


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def compute_borders(spectrogram, nb_intervals):
    """ Find interval edges and interval medians using median-cut """

    borders_per_mel_bin = []
    medians_per_mel_bin = []
    for mel_bin in range(spectrogram.shape[1]):

        # start with the complete spectrogram
        intervals = [(spectrogram.shape[0], spectrogram[:, mel_bin])]

        while len(intervals) < nb_intervals:

            # order intervals according to their sample length
            intervals = sorted(intervals, key=lambda x: x[0])

            # take longest interval data
            interval_data = intervals.pop()[1]

            # compute median index
            interval_data = np.sort(interval_data)
            median = interval_data[len(interval_data) // 2]

            # split interval at median
            left = interval_data[np.where(interval_data <= median)]
            right = interval_data[np.where(interval_data > median)]

            if len(left) > 0 and len(right) > 0:

                # append both intervals in the interval list
                intervals.append((len(left), left))
                intervals.append((len(right), right))
            else:

                # ignore this interval for further processing
                intervals.append((0, interval_data))

        # order intervals based in median value
        intervals = sorted(intervals, key=lambda x: np.median(x[1]))

        # extract right edge as borders and corresponding median of the interval
        interval_borders = [np.max(interval[1]) for interval in intervals]
        interval_medians = [np.median(interval[1]) for interval in intervals]

        # convert to numpy array
        interval_borders = np.array(interval_borders)
        interval_medians = np.array(interval_medians)

        borders_per_mel_bin.append(interval_borders)
        medians_per_mel_bin.append(interval_medians)

    # construct array of shape (40x8) containing borders
    borders_array = np.zeros((spectrogram.shape[1], nb_intervals))
    for i in range(len(borders_per_mel_bin)):
        borders_array[i, :] = borders_per_mel_bin[i]

    # construct array of shape (40x8) containing medians
    medians_array = np.zeros((spectrogram.shape[1], nb_intervals))
    for i in range(len(medians_per_mel_bin)):
        medians_array[i, :] = medians_per_mel_bin[i]

    return medians_array, borders_array


def compute_borders_logistic(spectrogram, nb_intervals):
    """ Find interval edges and interval medians using median-cut """

    vmins = np.min(spectrogram, axis=0)
    vmaxs = np.max(spectrogram, axis=0)

    def sigmoid(t, scaling, k=0.5):
        minimum, maximum = scaling
        L = abs(minimum) + maximum
        return L / (1 + np.exp(-k * t)) - abs(minimum)

    borders_array = np.zeros((spectrogram.shape[1], nb_intervals))
    medians_array = np.zeros((spectrogram.shape[1], nb_intervals))

    for bin in range(0, spectrogram.shape[1]):
        vmin = vmins[bin]
        vmax = vmaxs[bin]
        t = np.linspace(-10, 10, nb_intervals+1, endpoint=True)
        y = sigmoid(t, scaling=[vmin, vmax])
        borders_array[bin, :-1] = y[1:-1]
        borders_array[bin, -1] = vmax

        t = np.linspace(-9.5, 9.5, nb_intervals, endpoint=True)
        representative = sigmoid(t, scaling=[vmin, vmax])
        medians_array[bin, :] = representative

    return medians_array, borders_array


def quantize_spectrogram(spectrogram, borders):

    quantized_spectrogram = np.zeros(spectrogram.shape)
    for mel_bin in range(spectrogram.shape[1]):

        for interval_nb in reversed(range(borders.shape[1])):
            indices = np.where(spectrogram[:, mel_bin] <= borders[mel_bin, interval_nb])
            quantized_spectrogram[indices, mel_bin] = interval_nb

    # return to_categorical(quantized_spectrogram, num_classes=len(borders[0]))
    return quantized_spectrogram


def dequantize_spectrogram(q_spectrogram, medians_array):
    q_spectrogram = q_spectrogram.astype(int)
    spectrogram = np.zeros((q_spectrogram.shape[0], medians_array.shape[0]))

    for mel_bin in range(spectrogram.shape[1]):
        idx = q_spectrogram[:, mel_bin]
        med = medians_array[mel_bin]

        spectrogram[:, mel_bin] = med[idx]

    return spectrogram
