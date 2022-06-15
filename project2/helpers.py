from scipy import fftpack
from scipy import signal
import hrvanalysis
import heartpy as hp
from biosppy.signals import ecg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin

n_cpu = cpu_count()
print("Detected", n_cpu, "Logical CPU's for Parallel Processing.")

# setup for the feature names
tdf_names = [
    "mean_nni",
    "sdnn",
    "sdsd",
    "nni_50",
    "pnni_50",
    "nni_20",
    "pnni_20",
    "rmssd",
    "median_nni",
    "range_nni",
    "cvsd",
    "cvnni",
    "mean_hr",
    "max_hr",
    "min_hr",
    "std_hr",
]

gf_names = ["triangular_index"]

fdf_names = ["lf", "hf", "lf_hf_ratio", "lfnu", "hfnu", "total_power", "vlf"]

cscv_names = [
    "csi",
    "cvi",
    "Modified_csi",
]

pcp_names = ["sd1", "sd2", "ratio_sd2_sd1"]

samp_names = ["samp"]

hrv_names = np.concatenate(
    [tdf_names, gf_names, fdf_names, cscv_names, pcp_names, samp_names]
)


def median(templates, ax=0):
    """
    It takes the median of the templates along the axis specified by the ax argument

    :param templates: the array of templates
    :param ax: The axis along which the median is computed. The default is to compute the median along a
    flattened version of the array, defaults to 0 (optional)
    :return: The median of the templates.
    """
    return np.median(templates, axis=ax)


def mean(templates, ax=0):
    """
    It takes a list of templates and returns the mean of them

    :param templates: a list of numpy arrays, each of which is a template
    :param ax: The axis along which the mean is computed. The default is to compute the mean of the
    flattened array, defaults to 0 (optional)
    :return: The mean of the templates along the axis specified.
    """
    return np.mean(templates, axis=ax)


def mean_std(x):
    """
    It takes a list of numbers and returns the mean and standard deviation of the list

    :param x: The input data
    :return: The mean and standard deviation of the data.
    """
    return np.mean(x), np.std(x)


def generate_xlabels(temp):
    """
    It generates the x-axis labels for plotting the data

    :param temp: the data you want to plot
    :return: A list of floats.
    """
    timings = np.linspace(0, len(temp), 10)
    labels = []
    for item in timings:
        labels.append(round(item / 300, 2))
    return labels


def plot_pqrst(temp):
    """
    It plots the pqrst diagram
    :param temp: the data you want to plot
    """
    plt.plot(temp)
    plt.xticks(ticks=np.linspace(0, len(temp), 10), labels=generate_xlabels(temp))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")


def get_mvarmimad(temp):
    """
    It takes a list of numbers and returns its mean, standard deviation,
    minimum, maximum, and absolute difference between the maximum and minimum

    :param temp: the data to be analyzed
    :return: The mean, standard deviation, minimum, maximum, and delta of the data.
    """
    mean = np.mean(temp)
    std = np.std(temp)
    min = np.min(temp)
    max = np.max(temp)
    delta = np.abs(max - min)
    return [mean, std, min, max, delta]


def peak_enhancer(data, sr=300):
    """
    It takes in a list of ECG signals and returns a list of ECG signals with enhanced peaks

    :param data: The data to be processed
    :param sr: sample rate of the data, defaults to 300 (optional)
    :return: The peak-enhanced ECG
    """
    if len(np.shape(data)) > 1:
        return [hp.enhance_ecg_peaks(item, sample_rate=sr) for item in data]
    else:
        return hp.enhance_ecg_peaks(data, sample_rate=sr)


def get_ecg_data(data, sr=300, type="whole"):
    """
    It takes in a signal, a sampling rate, and a type of data you want to extract. The type is used
    as dictionary key for the ecg library ecg extraction. There is 'heart_rate', 'filtered', 'templates' or 'rpeaks' and 'whole' return everything.
    
    :param data: The data you want to analyze
    :param sr: sampling rate, defaults to 300 (optional)
    :param type: whole, heart_rate, filtered, templates, rpeaks, defaults to whole (optional)
    :return: The return value is a dictionary with the following keys:
    """
    if type == "whole":
        return ecg.ecg(signal=data, sampling_rate=sr)
    if type == "heart_rate":
        return ecg.ecg(signal=data, sampling_rate=sr, show=False)["heart_rate"]
    if type == "filtered":
        return ecg.ecg(signal=data, sampling_rate=sr, show=False)["filtered"]
    if type == "templates":
        return ecg.ecg(signal=data, sampling_rate=sr, show=False)["templates"]
    if type == "rpeaks":
        return ecg.ecg(signal=data, sampling_rate=sr, show=False)["rpeaks"]


# array used for frequency content, based on log spacing of an important
# frequency range in ECG
freq_array = [
    0.46981001,
    0.94989355,
    1.34722766,
    1.92037653,
    2.45140581,
    3.58897684,
    5.61198292,
    7.85416366,
    11.77742192,
    19.56926069,
]


def get_frequency_content(ecg, frequencies=freq_array):
    """
    It takes an ECG signal and returns the sum of the FFT in each of the frequency bands, the mean of
    the FFT, and the standard deviation of the FFT
    
    :param ecg: the ECG signal
    :param frequencies: The frequencies to be used for the frequency bands
    :return: the sum of the fft in the frequency bands, the mean and the standard deviation of the fft.
    """

    # calculate the fft and get the used frequencies with sampling rate of 300
    fft = fftpack.fft(ecg)
    freqs = fftpack.fftfreq(len(ecg)) * 300

    # get only the positive frequencies
    length = len(freqs[freqs >= 0])

    # cur at the crossover to negative frequencies, remove dc
    freqs = freqs[1:length]
    fft = np.abs(fft)[1:length]
    freqs = freqs[freqs <= frequencies[-1]]
    fft = fft[: len(freqs)]

    # calculate the sums in the frequency bands
    sums = []
    sums.append(np.sum(fft[freqs <= frequencies[0]]))
    for i in range(len(frequencies) - 1):
        sums.append(
            np.sum(fft[(freqs > frequencies[i]) & (freqs <= frequencies[i + 1])])
        )

    mean = np.average(fft, weights=freqs)
    std = np.sqrt(np.average((fft - mean) ** 2, weights=freqs))
    return np.concatenate((sums, mean, std), axis=None)


def get_heart_rate_data(ecg):
    """
    It takes the ECG data and returns the difference between the maximum and minimum heart rate and the
    variance of the heart rate
    
    :param ecg: the ECG data
    :return: a list of two values: the difference between the maximum and minimum heart rate and the
    variance of the heart rate.
    """
    rate = get_ecg_data(ecg, type="heart_rate")
    # not calculate values for low rates
    if len(rate) < 2:
        delta = np.NaN
        var = np.NaN
    else:
        delta = np.abs(np.max(rate) - np.min(rate))
        var = np.var(rate)
    return [delta, var]


def calc_rmssd(peaks):
    """
    It takes a list of peaks and returns the root mean square of the differences between each peak and
    the next peak
    
    :param peaks: a list of the R-R intervals in milliseconds
    :return: The root mean square of successive differences (RMSSD)
    """
    diff = [(peaks[i] - peaks[i + 1]) ** 2 for i in range(len(peaks) - 1)]
    return np.sqrt(np.mean(diff))

# pqrst feature names, used in the dataframe
pqrst_features = [
    "pr_mean",
    "pr_var",
    "ps_mean",
    "ps_var",
    "pt_mean",
    "pt_var",
    "qs_mean",
    "qs_var",
    "qt_mean",
    "qt_var",
    "rt_mean",
    "rt_var",
    "p_mean",
    "p_std",
    "q_mean",
    "q_std",
    "r_mean",
    "r_std",
    "s_mean",
    "s_std",
    "t_mean",
    "t_std",
]


def calc_pqrst_data(ecg):
    """
    It takes the templates from the ECG data and calculates the mean and standard deviation of the time
    between the P, Q, R, S, and T waves
    
    :param ecg: the ecg data
    :return: a numpy array of the mean and standard deviation of the following:
    """
    templates = get_ecg_data(ecg, type="templates")
    pr = []
    ps = []
    pt = []
    qs = []
    qt = []
    rt = []
    p_vals = []
    q_vals = []
    r_vals = []
    s_vals = []
    t_vals = []
    # extract the pqrst indexes for each template (note that the indexes are sampled at 300Hz!)
    for template in templates:
        # calculate the locations
        try:
            # get local maximas and minimas with signal library
            loc_max = np.array(signal.argrelextrema(template, np.greater))
            loc_min = np.array(signal.argrelextrema(template, np.less))
            # find the maximum, but cut the search area to the first half to avoid
            # finding peaks at the wrong place
            r = np.argmax(template[: int(len(template) / 2)])
            # q and s are the first minima after and before the r value
            q = loc_min[loc_min < r][-1]
            s = loc_min[loc_min > r][0]
            # p and t are the first maxima after and before the r value
            p = loc_max[loc_max < r][-1]
            t = loc_max[loc_max > r][0]
            p_vals.append(template[p])
            q_vals.append(template[q])
            r_vals.append(template[r])
            s_vals.append(template[s])
            t_vals.append(template[t])
            # calculate different values
            pr.append(r - p)
            ps.append(s - p)
            pt.append(t - p)
            qs.append(s - q)
            qt.append(t - q)
            rt.append(t - r)
        except:
            pr.append(0)
            ps.append(0)
            pt.append(0)
            qs.append(0)
            qt.append(0)
            rt.append(0)

    # calculate different additional data, there will be the
    # mean and the std and the delta (other values were bad)
    pr = mean_std(pr)
    ps = mean_std(ps)
    pt = mean_std(pt)
    qs = mean_std(qs)
    qt = mean_std(qt)
    rt = mean_std(rt)
    # calculate the mean and var amplitudes
    p_vals = mean_std(p_vals)
    q_vals = mean_std(q_vals)
    r_vals = mean_std(r_vals)
    s_vals = mean_std(s_vals)
    t_vals = mean_std(t_vals)
    return np.concatenate(
        [pr, ps, pt, qs, qt, rt, p_vals, q_vals, r_vals, s_vals, t_vals]
    )


def get_peaks_data(ecg, thresh=0.6):
    """
    It takes an ECG signal and returns a list of 8 features, extracted from peaks: mean, min, max, rmssd, sdnn, delta,
    thresh_rate, thresh_over_peak
    
    :param ecg: the ecg data
    :param thresh: the threshold for the threshold_rate parameter
    :return: The mean, min, max, rmssd, sdnn, delta, thresh_rate, thresh_over_peak
    """
    peaks = get_ecg_data(ecg, type="rpeaks")
    num = len(peaks)
    # if the array is too short, return a NaN array
    if num < 2:
        print("Peak Array length too short!")
        return [0, 0, 0, 0, 0, 0, 0]

    # calculate the intervals from the peak positions
    diff = get_rdiff(peaks)
    # get basic mean(ibi), variance and delta of the max to min
    mean, var, min, max, delta = get_mvarmimad(diff)
    # calculate the rmssd
    rmssd = calc_rmssd(diff)
    # calculate the sdnn
    sdnn = np.sqrt(var)
    # calculate the time where the sample is above a certain threshold
    thres = np.max(ecg) * thresh
    thresh_rate = sum(ecg > thres) / len(ecg)
    thresh_over_peak = thresh_rate / len(peaks)

    return [mean, min, max, rmssd, sdnn, delta, thresh_rate, thresh_over_peak]


def filter_and_remove_baseline(x):
    """
    It removes the baseline wander from the signal and then filters it using a low pass filter with a
    cutoff frequency of 30Hz
    
    :param x: the signal to be filtered
    :return: The filtered signal
    """

    x = hp.remove_baseline_wander(x, sample_rate=300)
    # notch around 50hz against power noise
    x = hp.filter_signal(x, sample_rate=300, cutoff=30, filtertype="lowpass", order=3)
    return x


def preprocess(dataset):
    proc = Parallel(n_jobs=n_cpu)(delayed(filter_and_remove_baseline)(i) for i in dataset)
    return proc


def get_class_mean(classdata):
    return np.mean([np.mean(item) for item in classdata])


def get_class_median(classdata):
    return np.median([np.median(item) for item in classdata])


def get_rdiff(rp):
    diff = []
    for idx in range(len(rp) - 1):
        diff.append(rp[idx + 1] - rp[idx])
    return diff


def get_hp_data(ecg):
    """
    It takes an ECG signal and returns the pnn20, pnn50, sd1, sd2 and s, calculated in heartpy
    
    :param ecg: The ECG signal
    :return: a list of 6 features.
    """
    try:
        _, measures = hp.process(ecg, sample_rate=300)
    except:
        print("Normal HP measures failed")
        pass
    try:
        _, measures = hp.process(hp.flip_signal(ecg), sample_rate=300)
    except:
        print("HP failed")
        return [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
    # check if s is nan, flip signal if thats the case
    if measures["s"] != measures["s"]:
        _, measures = hp.process(hp.flip_signal(ecg), sample_rate=300)

    features = []
    features.append(measures["pnn20"])
    features.append(measures["pnn50"])
    features.append(measures["sd1"])
    features.append(measures["sd2"])
    features.append(measures["s"])
    features.append(np.log10(measures["sd1/sd2"] ** 2))
    return features


def get_hrv_data(ecg):
    """
    It takes in an ECG signal and returns a list of HRV features
    
    :param ecg: the ECG data
    :return: The return value is a list of all the HRV features.
    """
    ret = []
    rpeaks = get_ecg_data(ecg, type="rpeaks")
    tdf = hrvanalysis.get_time_domain_features(rpeaks)
    gf = hrvanalysis.get_geometrical_features(rpeaks)
    fdf = hrvanalysis.get_frequency_domain_features(rpeaks)
    cscv = hrvanalysis.get_csi_cvi_features(rpeaks)
    pcp = hrvanalysis.get_poincare_plot_features(rpeaks)
    samp = hrvanalysis.get_sampen(rpeaks)

    for name in tdf_names:
        ret.append(tdf[name])

    for name in gf_names:
        ret.append(gf[name])

    for name in fdf_names:
        ret.append(fdf[name])

    for name in cscv_names:
        ret.append(cscv[name])

    for name in pcp_names:
        ret.append(pcp[name])

    ret.append(samp["sampen"])

    return ret


def get_template_features(ecg):
    templates = get_ecg_data(ecg, type="templates")
    med_template = median(templates, ax=0)
    std_sum = sum(np.std(templates, axis=0))
    mean, var, _, _, delta = get_mvarmimad(med_template)
    return mean, var, delta, std_sum


def signaltonoise_dB(a, axis=0, ddof=0):
    """
    The function takes an array, calculates the mean and standard deviation, and returns the
    signal-to-noise ratio in decibels
    
    :param a: array_like
    :param axis: axis along which to compute the mean and standard deviation, defaults to 0 (optional)
    :param ddof: Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N
    represents the number of elements. By default ddof is zero, defaults to 0 (optional)
    :return: The signal-to-noise ratio (SNR) in decibels (dB).
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))


def fix_flipped(signal):
    """
    If the mean of the signal is less than 0.07 and the absolute value of the minimum of the signal is
    greater than 1.6 times the maximum of the signal, then flip the signal
    
    :param signal: the signal to be flipped
    :return: The signal is being returned.
    """
    if np.mean(signal) < 0.07 and 1.6 * max(signal) < abs(min(signal)):
        return hp.flip_signal(signal)
    else:
        return signal


def extract_features(dataset):
    frame = pd.DataFrame()

    print("Extracting data...")
    # flip signals if necessary
    dataset = Parallel(n_jobs=n_cpu)(delayed(fix_flipped)(i) for i in dataset)
    # center the data because it was flipped and is not centered in general
    dataset = Parallel(n_jobs=n_cpu)(
        delayed(get_ecg_data)(i, type="filtered") for i in dataset
    )
    # calculate all the different features in parralel
    freq_prop = Parallel(n_jobs=n_cpu)(
        delayed(get_frequency_content)(i) for i in dataset
    )
    mvarmimad = Parallel(n_jobs=n_cpu)(delayed(get_mvarmimad)(i) for i in dataset)
    rate_prop = Parallel(n_jobs=n_cpu)(delayed(get_heart_rate_data)(i) for i in dataset)
    peaks_prop = Parallel(n_jobs=n_cpu)(delayed(get_peaks_data)(i) for i in dataset)
    hp_prop = Parallel(n_jobs=n_cpu)(delayed(get_hp_data)(i) for i in dataset)
    hrv_prop = Parallel(n_jobs=n_cpu)(delayed(get_hrv_data)(i) for i in dataset)

    template_prop = Parallel(n_jobs=n_cpu)(
        delayed(get_template_features)(i) for i in dataset
    )
    pqrst_prop = Parallel(n_jobs=n_cpu)(delayed(calc_pqrst_data)(i) for i in dataset)

    print("Extracting successful, transforming to DataFrame.")
    # convert to df columns
    mvarmimad = pd.DataFrame(mvarmimad, columns=["mean", "var", "min", "max", "delta"])
    rate_prop = pd.DataFrame(rate_prop, columns=["delta_hr", "var_hr"])
    peaks_prop = pd.DataFrame(
        peaks_prop,
        columns=[
            "mean_rp",
            "min_rp",
            "max_rp",
            "rmssd_rp",
            "sdnn_rp",
            "delta_rp",
            "thresh_rate",
            "thresh/peaks",
        ],
    )
    hp_prop = pd.DataFrame(
        hp_prop,
        columns=["hp_pnn20", "hp_pnn50", "hp_sd1", "hp_sd2", "hp_s", "hp_s1_over_s2"],
    )
    hrv_prop = pd.DataFrame(hrv_prop, columns=hrv_names)
    template_prop = pd.DataFrame(
        template_prop,
        columns=[
            "template_mean",
            "template_var",
            "template_delta",
            "template_integrated_std",
        ],
    )
    freq_prop = pd.DataFrame(
        freq_prop,
        columns=[
            "fbin1",
            "fbin2",
            "fbin3",
            "fbin4",
            "fbin5",
            "fbin6",
            "fbin7",
            "fbin8",
            "fbin9",
            "fbin10",
            "fmean",
            "fstd",
        ],
    )
    # pqrst extraction, there is a lot!
    pqrst_prop = pd.DataFrame(pqrst_prop, columns=pqrst_features)
    frame = pd.concat(
        [
            frame,
            mvarmimad,
            rate_prop,
            peaks_prop,
            template_prop,
            pqrst_prop,
            hrv_prop,
            hp_prop,
            freq_prop,
        ],
        axis=1,
    )
    return frame


def extract_noise_features(dataset):
    frame = pd.DataFrame()
    snr = []

    for item in dataset:
        snr.append(signaltonoise_dB(item))

    snr = pd.DataFrame(snr, columns=["snr"])
    frame = pd.concat([frame, snr], axis=1)
    return frame


def read_process(path):
    X = pd.read_csv(path)
    X.drop(columns="id", inplace=True)
    col_names = X.index
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = pd.DataFrame(scaler.fit_transform(X.transpose()).transpose())
    X = [item[~np.isnan(item)] for item in X.to_numpy()]
    X_noise = extract_noise_features(X)
    X = preprocess(X)
    X = extract_features(X)
    X = pd.concat([X, X_noise], axis=1)
    return X, col_names


def correlation_remover(dataset, threshold):
    reduced = dataset.copy()
    col_corr = set()
    corr_matrix = reduced.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (
                corr_matrix.columns[j] not in col_corr
            ):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in reduced.columns:
                    del reduced[colname]
    return reduced, col_corr


def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif


class ExperimentalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X)
        self.yhat = self.model.predict(X)
        return self

    def transform(self, X, y):
        X_ = X.copy()
        y_ = y.copy()
        return X_.iloc[np.array(self.yhat == -1), :], y_.iloc[np.array(self.yhat == -1)]
