import numpy
from scipy.fftpack import dct
import scipy.fft as sp

def MFCC_Features(signal):
    samplerate = 8000
    window_length = 0.025
    window_step = 0.01
    cepstr_number = 13
    cep_lifter = 22
    # Calculate nfft
    samples_length = window_length * samplerate
    nfft = 1
    while nfft < samples_length:
        nfft *= 2
    # print(nfft)

    mfcc_feats, total_energy = filter_bank(signal, samplerate, window_length, window_step, nfft)
    mfcc_feats = numpy.log(mfcc_feats)
    mfcc_feats = sp.dct(mfcc_feats, type=2, axis=1, norm='ortho')[:, :cepstr_number]
    mfcc_feats = lifter(mfcc_feats, cep_lifter)
    mfcc_feats[:, 12] = numpy.log(total_energy)  # replace first cepstral coefficient with log of frame energy
    delta_features = delta(mfcc_feats, 13)
    double_delta_features = double_delta(mfcc_feats, 13)

    # mfcc_feats = numpy.c_[mfcc_feats, delta_features]
    # mfcc_feats = numpy.c_[mfcc_feats, double_delta_features]
    return mfcc_feats


def lifter(cepstral, cep_filters):
    if cep_filters > 0:
        number_frames, n_coeffs = numpy.shape(cepstral)
        n = numpy.arange(n_coeffs)
        lift = 1 + (cep_filters / 2.) * numpy.sin(numpy.pi * n / cep_filters)
        return lift * cepstral
    else:
        return cepstral


def filter_bank(signal, samplerate, window_length, window_step, nfft):
    signal = preemphasis(signal)
    frames = window(signal, samplerate)
    power_spec = power_spectogram(frames, nfft)
    total_energy = numpy.sum(power_spec, 1)
    total_energy = numpy.where(total_energy == 0, numpy.finfo(float).eps, total_energy)

    n_filter = 40
    filter_banks = get_filter_banks(n_filter, nfft, samplerate)
    features = numpy.dot(power_spec, filter_banks.T)  # filterbank energy
    features = numpy.where(features == 0, numpy.finfo(float).eps, features)

    return features, total_energy


def get_filter_banks(n_filter, nfft, samplerate):
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (samplerate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, n_filter + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((nfft + 1) * hz_points / samplerate)

    filter_banks = numpy.zeros((n_filter, int(numpy.floor(nfft / 2 + 1))))
    for m in range(1, n_filter + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            filter_banks[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            filter_banks[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return filter_banks


def power_spectogram(frames, nfft):
    # Magnitides
    fft_frames = numpy.absolute(sp.rfft(frames, nfft))
    # Power Spec
    pw_frames = ((1.0 / nfft) * (fft_frames ** 2))
    return pw_frames


def window(emphasized_signal, sample_rate):
    # Windowing and Framing

    # Typical frame sizes 20-40 ms
    # 10-15 ms stride(overlap)

    frame_size = 0.025
    frame_overlap = 0.01

    frame_length, frame_step = frame_size * sample_rate, frame_overlap * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    frames_number = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = frames_number * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)

    indices = numpy.tile(numpy.arange(0, frame_length), (frames_number, 1)) + numpy.tile(
        numpy.arange(0, frames_number * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    frames *= numpy.hamming(frame_length)
    return frames


def preemphasis(signal):
    # Pre-Emphasis
    pre_emphasis = 0.97

    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal


def delta(features, N):
    if N < 1:
        raise ValueError('N must be int >= 1')
    # NUMBER_OF_FRAMES = len(features)
    # denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    # delta_features = numpy.empty_like(features)
    delta_features = numpy.zeros(features.shape)
    NUMBER_OF_FRAMES = features.shape[1]
    padded = numpy.pad(features, ((N, N), (0, 0)), mode='edge')

    for t in range(NUMBER_OF_FRAMES):
        minus_one, plus_one = t-1, t+1
        if minus_one < 0:
            minus_one = 0
        if plus_one >= NUMBER_OF_FRAMES:
            plus_one = NUMBER_OF_FRAMES - 1
        delta_features[:, t] = 0.5*(features[:, plus_one]-features[:, minus_one])
    return delta_features

    #     delta_features[t] = numpy.dot(numpy.arange(-N, N + 1), padded[t: t + 2 * N + 1]) / denominator
    # return delta_features

def double_delta(features, N):
    double_delta_features = numpy.zeros(features.shape)
    NUMBER_OF_FRAMES = features.shape[1]
    for t in range(NUMBER_OF_FRAMES):
        minus_two, minus_one, plus_one, plus_two = t-2, t-1, t+1, t+2

    if minus_one < 0:
        minus_one = 0
    if plus_one >= NUMBER_OF_FRAMES:
        plus_one = NUMBER_OF_FRAMES - 1
    if minus_two < 0:
        minus_two = 0
    if plus_two >= NUMBER_OF_FRAMES:
        plus_two = NUMBER_OF_FRAMES - 1

    double_delta_features[:, t] = 0.1*(2*features[:, plus_two] + features[:, plus_one] - features[:, minus_one]-2*features[:, minus_two])
    return double_delta_features

