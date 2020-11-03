import glob
import numpy as np
import scipy as sp
import scipy.io
import scipy.signal


def LoadTroikaDataset():
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the
            reference data for data_fls[5], etc...
    """
    data_dir = "part_1/datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls


def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]


def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability.

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]

    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))


fs = 125  # signals were sampled at 125 Hz
minBPM = 40  # min bpm
maxBPM = 240  # max bpm
window_length = 8 * fs  # 8 second time window
window_shift = 2 * fs  # 2 second shift to next window


def bandpass_filter(signal, fs):
    """filter the signal between 40 and 240 BPM

    Args:
        signal ([np_array]): input signal
        fs ([int]): Hz of input signal

    Returns:
        [np_array]: filtered signal
    """
    pass_band = (minBPM/60, maxBPM/60)
    b, a = scipy.signal.butter(3, pass_band, btype='bandpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def fourier_transform(signal, fs):
    """compute and return the one-dimensional fourier transform
    and the fourier transformed frequencies

    Args:
        signal (np_array): input signal
        fs (int): Hz of input signal

    Returns:
        fft (np_array): one-dimensional fourier transform
        freqs (np_array): fourier transformed frequencies
    """
    fft = np.abs(np.fft.rfft(signal, 2*len(signal)))
    freqs = np.fft.rfftfreq(2*len(signal), 1/fs)
    return fft, freqs


def calculate_confidence(freqs, fft_f, bpm_max):
    """calculates the confidence value for a signal window

    Args:
        freqs (np_array): list of frequenqies
        fft_f (np_array): fourier transformed signal
        bpm_max (float): max frequency

    Returns:
        confidence value (float64)
    """
    fundamental_freq_window = (
        freqs > bpm_max - minBPM/60) & (freqs < bpm_max + minBPM/60)
    return np.sum(fft_f[fundamental_freq_window]) / np.sum(fft_f)


def RunPulseRateAlgorithm(data_fl, ref_fl):
    """Handler function for computing the pulse rate

    Args:
        data_fl (.mat file ): ppg and acc data
        ref_fl (.mat file): ground truth data

    Returns:
        errors (np_array): difference between ground truth and predictions
        confidence (np_array): confidence values for heart rate predictions
    """
    # load data using LoadTroikaDataFile
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)

    # loading the reference file
    ground_truth = sp.io.loadmat(ref_fl)['BPM0']

    # bandpass filter the signals
    ppg = bandpass_filter(ppg, fs)
    accx = bandpass_filter(accx, fs)
    accy = bandpass_filter(accy, fs)
    accz = bandpass_filter(accz, fs)

    # list of the estimated bpms
    bpm_pred = []

    # list of the calculated confidence values
    confidence = []

    # analyze a single window of ppg and acc data
    # and compute a bpm prediction and a confidence value
    for i in range(0, len(ppg) - window_length, window_shift):
        ppg_window = ppg[i:i+window_length]

        # aggregate accelerometer data into single signal to get the acc window
        acc_window = np.sqrt(accx**2 + accy**2 + accz**2)
        acc_window = acc_window[i:i+window_length]

        # fft the ppg and acc signals
        fft_ppg, ppg_freqs = fourier_transform(ppg_window, fs)
        fft_acc, acc_freqs = fourier_transform(acc_window, fs)

        # filter the signals
        fft_ppg[ppg_freqs <= (minBPM + 10)/60.0] = 0.0
        fft_ppg[ppg_freqs >= (maxBPM - 10)/60.0] = 0.0

        fft_acc[acc_freqs <= (minBPM + 10)/60.0] = 0.0
        fft_acc[acc_freqs >= (maxBPM - 10)/60.0] = 0.0

        # get the maximum value of the ppg and acc signal
        ppg_max = ppg_freqs[np.argsort(fft_ppg, axis=0)[-1]]
        acc_max = acc_freqs[np.argsort(fft_acc, axis=0)[-1]]
        ppg_max_2 = ppg_freqs[np.argsort(fft_ppg, axis=0)[-2]]
        acc_max_2 = acc_freqs[np.argsort(fft_acc, axis=0)[-2]]

        if ppg_max_2 < ppg_max:
            ppg_max = ppg_max_2

        if acc_max_2 < acc_max:
            acc_max = acc_max_2

        conf_val = calculate_confidence(ppg_freqs, fft_ppg, ppg_max)
        bpm_pred.append(ppg_max*60)
        confidence.append(conf_val)

    # Return per-estimate mean absolute error and confidence as a 2-tuple of numpy arrays.
    errors = np.abs(np.diag(np.subtract(ground_truth, bpm_pred)))
    return errors, confidence


def Evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs = []
    confs = []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
    # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return AggregateErrorMetric(errs, confs)
