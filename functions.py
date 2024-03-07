import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import mode
from scipy.stats import pearsonr
from scipy.stats import describe
from scipy.stats import zscore
import biosppy.signals.eda as eda
import biosppy.signals.resp as resp
from biosppy.signals import ecg
import pyhrv.tools as tools
from pyhrv.hrv import hrv


def segment_data(data, sampling_rate=700, segment_length=60, window_stride=0.25):
    """
    Segment the given data array into 60-second segments with a sliding window of 0.25 seconds.
    
    Args:
    - data: The data array.
    - sampling_rate: The sampling rate of the data in Hz (default: 700 Hz).
    - segment_length: The length of each segment in seconds (default: 60 seconds).
    - window_stride: The stride of the sliding window in seconds (default: 0.25 seconds).
    
    Returns:
    - segments: A list of segmented data arrays.
    """
    
    segments = []
    start_index = 0
    end_index = int(segment_length * sampling_rate)
    window_stride_samples = int(window_stride * sampling_rate)
    
    while end_index <= len(data):
        segments.append(data[start_index:end_index])
        start_index += window_stride_samples
        end_index += window_stride_samples
    
    return segments


def segment_labels(labels, sampling_rate=700, segment_length=60, window_stride=0.25):
    """
    Segment the labels into 2-second segments with a sliding window of 0.25 seconds.
    
    Args:
    - labels: The labels array.
    - sampling_rate: The sampling rate of the data in Hz (default: 700 Hz).
    - segment_length: The length of each segment in seconds (default: 60 seconds).
    - window_stride: The stride of the sliding window in seconds (default: 0.25 seconds).
    
    Returns:
    - seg_labels: A list containing the majority label for each segment.
    - label_fractions: A list indicating what fraction of the segment the label 
    applies to (maybe useful if a label boundary falls in the middle of the segment)
    """

    seg_labels = []
    label_fractions = []
    start_index = 0
    end_index = int(segment_length * sampling_rate)
    window_stride_samples = int(window_stride * sampling_rate)
    
    while end_index <= len(labels):
        seg_label, label_count = mode(labels[start_index:end_index])
        seg_labels.append(seg_label)
        label_fractions.append(label_count/(end_index-start_index))
        start_index += window_stride_samples
        end_index += window_stride_samples
    
    return seg_labels, label_fractions


def remove_unused_segments(original_dict):
    '''
    Keep only segments that correspond to a valid label (label = 1, 2, or 3)
    where that label is valid for the entire segment (label_fracs = 1).
    '''
    
    labels = np.array(original_dict['labels'])
    label_fracs = np.array(original_dict['label_fracs'])
    mask = (np.isin(labels, [1, 2, 3]) & (label_fracs == 1))
    
    valid_segments_dict = {}
    for data_type in original_dict:
        segments = np.array(original_dict[data_type])
        valid_segments_dict[data_type] = segments[mask]
    
    return valid_segments_dict


## Feature extraction

def get_peak_frequency(acceleration_data, sampling_rate):
    # Perform FFT
    fft_result = np.fft.fft(acceleration_data)
    # Frequency bins
    freq_bins = np.fft.fftfreq(len(acceleration_data), d=1/sampling_rate)
    # Find magnitude spectrum
    magnitude_spectrum = np.abs(fft_result)
    # Find index of peak frequency
    peak_freq_index = np.argmax(magnitude_spectrum)
    # Calculate peak frequency
    peak_frequency = freq_bins[peak_freq_index]
    return peak_frequency


def get_ACC_features(data):
    """
    Calculate the mean and STD for each axis separately and summed across all axes. 
    Calculate the absolute integral for each axis.
    Calculate the peak frequency for each axis. 
    """

    axis0 = data[:,0]
    axis1 = data[:,1]
    axis2 = data[:,2]
    
    acc_features = [np.mean(axis0), np.mean(axis1), np.mean(axis2), np.mean(data),
                np.std(axis0), np.std(axis1), np.std(axis2), np.std(data),
               np.sum(np.abs(axis0)), np.sum(np.abs(axis1)), np.sum(np.abs(axis2)),
               get_peak_frequency(axis0, 700), get_peak_frequency(axis1, 700), get_peak_frequency(axis2, 700)]
    acc_feature_names = ['ACC x mean', 'ACC y mean', 'ACC z mean', 'ACC mean', 'ACC x std', 'ACC y std', 
                         'ACC z std', 'ACC std', 'ACC abs integral x', 'ACC abs integral y', 'ACC abs integral z', 
                         'ACC peak frequency x', 'ACC peak frequency y', 'ACC peak frequency z']
    return acc_features, acc_feature_names


def get_EMG_features(emg_signal):
    '''
    Calculate mean, STD, median, dynamic range, and absolute integral of EMG signal.
    Calculate 10th and 90th percentile.
    Calculate mean, median and peak frequency and energy in 7 bands.
    Calculate # peaks and mean, STD of peak amplitudes.
    Calculate sum and normalised sum of peak amplitudes.
    '''
    
    mean_emg = np.mean(emg_signal)
    std_emg = np.std(emg_signal)
    median_emg = np.median(emg_signal)
    dynamic_range = np.max(emg_signal) - np.min(emg_signal)
    abs_integral = np.sum(np.abs(emg_signal))
    percentile_10 = np.percentile(emg_signal, 10)
    percentile_90 = np.percentile(emg_signal, 90)
    
    # Frequency analysis
    frequencies, power_spectrum = signal.welch(emg_signal, fs=700)
    mean_frequency = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    median_frequency = np.median(frequencies)
    peak_frequency = frequencies[np.argmax(power_spectrum)]
    
    # Energy in seven bands - TODO
    '''
    energy_bands = []
    for band in range(7):
        lower_freq = 10 * band
        upper_freq = 10 * (band + 1)
        band_energy = np.sum(power_spectrum[(frequencies >= lower_freq) & (frequencies < upper_freq)])
        energy_bands.append(band_energy)
    '''
    
    # Calculate number of peaks, mean and standard deviation of peak amplitudes
    peaks, _ = signal.find_peaks(emg_signal.flatten())
    num_peaks = len(peaks)
    peak_amplitudes = emg_signal[peaks]
    mean_peak_amplitude = np.mean(peak_amplitudes)
    std_peak_amplitude = np.std(peak_amplitudes)
    
    # Calculate sum and normalized sum of peak amplitudes
    sum_peak_amplitudes = np.sum(peak_amplitudes)
    normalized_sum_peak_amplitudes = sum_peak_amplitudes / dynamic_range
    
    emg_features = [mean_emg, std_emg, median_emg, dynamic_range, abs_integral,
                   percentile_10, percentile_90, mean_frequency, median_frequency,
                   peak_frequency, num_peaks, mean_peak_amplitude,
                   std_peak_amplitude, sum_peak_amplitudes, normalized_sum_peak_amplitudes]
    emg_feature_names = ['EMG mean', 'EMG std', 'EMG median', 'EMG dynamic range', 'EMG absolute integral',
                     'EMG 10th percentile', 'EMG 90th percentile', 'EMG mean frequency', 'EMG median frequency',
                     'EMG peak frequency', 'EMG # peaks', 'EMG mean peak amplitude', 'EMG std peak amplitude', 
                     'EMG sum peak amplitudes', 'EMG normalized sum peak amplitudes']
    return emg_features, emg_feature_names


def get_resp_features(resp_signal):
    
    # Process respiration signal
    processed_resp = resp.resp(signal=resp_signal, sampling_rate=700, show=False)
    
    # Calculate inhalation/exhalation durations
    inhalation_durations = processed_resp['filtered'][processed_resp['filtered'] > 0]
    exhalation_durations = processed_resp['filtered'][processed_resp['filtered'] < 0]
    
    mean_inhalation_duration = np.mean(inhalation_durations)
    std_inhalation_duration = np.std(inhalation_durations)
    mean_exhalation_duration = np.mean(exhalation_durations)
    std_exhalation_duration = np.std(exhalation_durations)
    inhalation_exhalation_ratio = np.mean(np.abs(inhalation_durations)) / np.mean(np.abs(exhalation_durations))
    resp_range = np.max(processed_resp['filtered']) - np.min(processed_resp['filtered'])
    breath_rate = 60 / (len(processed_resp['filtered']) / 700)
    respiration_duration = len(processed_resp['filtered']) / 700
    
    resp_features = [mean_inhalation_duration, std_inhalation_duration, 
                mean_exhalation_duration, std_exhalation_duration,
                inhalation_exhalation_ratio, resp_range, 
                breath_rate, respiration_duration]
    resp_feature_names = ['mean inhale duration', 'std inhale duration', 'mean exhale duration', 'std exhale duration',
                          'inhale exhale ratio', 'resp range', 'breath rate', 'resp duration']
    return resp_features, resp_feature_names


def get_temp_features(temp_signal):
    '''
    Calculate the mean, std, min, max, range, and slope of the temp data.
    '''
    
    mean_temp = np.mean(temp_signal)
    std_temp = np.std(temp_signal)
    min_temp = np.min(temp_signal)
    max_temp = np.max(temp_signal)
    dynamic_range = max_temp - min_temp
    mean_slope = np.mean(np.diff(temp_signal))
    
    temp_features = [mean_temp, std_temp, min_temp, max_temp,
                    dynamic_range, mean_slope]
    temp_feature_names = ['temp mean', 'temp std', 'temp min', 'temp max', 'temp dynamic range', 'temp slope']
    return temp_features, temp_feature_names
    
 
