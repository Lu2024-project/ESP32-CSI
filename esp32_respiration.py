# -*-coding:utf-8-*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft
from pathlib import Path
from PyEMD import EMD


def emd_and_rebuild(s):
    """EMD分解和重建函数 - 更保守的处理"""
    try:
        emd = EMD()
        imf_a = emd.emd(s)

        # 更保守的重建 - 只去除最高频的噪声分量
        if len(imf_a) > 2:
            # 保留除了第一个最高频分量外的所有分量
            new_s = np.sum(imf_a[1:], axis=0)
        else:
            new_s = s

        # 检查重建信号的能量
        orig_energy = np.sum(s ** 2)
        new_energy = np.sum(new_s ** 2)
        if new_energy < orig_energy * 0.1:  # 如果能量损失过大，返回原始信号
            return s

        return new_s
    except Exception as e:
        print(f"EMD processing failed: {e}")
        return s


def calib(phase, k, axis=1):
    # 相位校准

    p = np.asarray(phase)
    k = np.asarray(k)

    slice1 = [slice(None, None)] * p.ndim
    slice1[axis] = slice(-1, None)
    slice1 = tuple(slice1)
    slice2 = [slice(None, None)] * p.ndim
    slice2[axis] = slice(None, 1)
    slice2 = tuple(slice2)
    shape1 = [1] * p.ndim
    shape1[axis] = k.shape[0]
    shape1 = tuple(shape1)

    k_n, k_1 = k[-1], k[0]
    a = (p[slice1] - p[slice2]) / (k_n - k_1)
    b = p.mean(axis=axis, keepdims=True)
    k = k.reshape(shape1)

    phase_calib = p - a * k - b
    return phase_calib


def advanced_peak_detection(signal_data, fs, min_distance=None):
    # 改进的峰值检测算法，更适应不同频率的呼吸信号

    if min_distance is None:
        # 更灵活的最小距离设置
        min_distance = max(int(fs * 60 / 50), 2)  # 对应最大50 BPM

    # 信号预处理 - 增强峰值
    signal_smooth = signal.savgol_filter(signal_data, window_length=5, polyorder=2)

    # 计算信号的动态特性
    signal_std = np.std(signal_smooth)
    signal_median = np.median(np.abs(signal_smooth))

    # 自适应阈值 - 降低以提高敏感度
    height_threshold = signal_median + 0.3 * signal_std

    # 多尺度峰值检测
    peaks_list = []

    # 尺度1：标准检测
    peaks1, _ = signal.find_peaks(
        signal_smooth,
        height=height_threshold,
        distance=min_distance,
        prominence=signal_std * 0.2
    )
    peaks_list.extend(peaks1)

    # 尺度2：更敏感的检测（用于高频呼吸）
    peaks2, _ = signal.find_peaks(
        signal_smooth,
        height=signal_median + 0.1 * signal_std,
        distance=max(int(min_distance * 0.7), 2),
        prominence=signal_std * 0.1
    )
    peaks_list.extend(peaks2)

    # 去重并排序
    peaks = np.unique(peaks_list)

    return peaks


def enhanced_fft_analysis(signal_data, fs):
    # 增强的FFT频域分析，改进对不同频率呼吸的检测能力

    n = len(signal_data)

    # 应用窗函数减少频谱泄漏
    windowed_signal = signal_data * signal.windows.hann(n)

    # FFT分析
    fft_result = np.abs(np.fft.fft(windowed_signal, n=n*2))  # 零填充提高频率分辨率
    freqs = np.fft.fftfreq(n*2, 1/fs)

    # 只考虑正频率
    positive_freqs = freqs[:n]
    positive_fft = fft_result[:n]

    # 扩大呼吸频率范围：0.15-0.6Hz (9-36 BPM)
    respiration_mask = (positive_freqs >= 0.15) & (positive_freqs <= 0.6)
    respiration_freqs = positive_freqs[respiration_mask]
    respiration_fft = positive_fft[respiration_mask]

    if len(respiration_freqs) == 0:
        return 0, 0

    # 平滑频谱
    if len(respiration_fft) > 5:
        respiration_fft_smooth = signal.savgol_filter(respiration_fft,
                                                      window_length=min(5, len(respiration_fft)//2*2+1),
                                                      polyorder=1)
    else:
        respiration_fft_smooth = respiration_fft

    # 找到主峰频率
    max_idx = np.argmax(respiration_fft_smooth)
    dominant_freq = respiration_freqs[max_idx]
    peak_power = respiration_fft_smooth[max_idx]

    # 寻找次要峰值（可能的谐波或其他呼吸模式）
    # 找到所有局部最大值
    local_peaks, properties = signal.find_peaks(
        respiration_fft_smooth,
        height=peak_power * 0.3,  # 至少是主峰的30%
        distance=max(1, len(respiration_fft_smooth)//20)
    )

    # 获取候选频率
    candidate_freqs = []
    candidate_powers = []

    for peak_idx in local_peaks:
        freq = respiration_freqs[peak_idx]
        power = respiration_fft_smooth[peak_idx]
        bpm = freq * 60

        # 只考虑合理的呼吸频率
        if 5 <= bpm <= 45:
            candidate_freqs.append(freq)
            candidate_powers.append(power)

    if not candidate_freqs:
        bpm = dominant_freq * 60
        return bpm if 5 <= bpm <= 45 else 0, peak_power

    # 选择最强的候选频率
    best_idx = np.argmax(candidate_powers)
    best_freq = candidate_freqs[best_idx]
    best_power = candidate_powers[best_idx]

    bpm = round(best_freq * 60)
    return bpm, best_power


def improved_peak_rate_calculation(peaks, fs):
    # 改进的基于峰值间隔的呼吸率计算

    if len(peaks) < 3:  # 至少需要3个峰值才能计算
        return 0, 0

    # 计算所有峰值间隔
    intervals = np.diff(peaks) / fs  # 转换为秒

    # 过滤异常间隔
    valid_intervals = intervals[(intervals >= 60/45) & (intervals <= 60/5)]  # 5-45 BPM

    if len(valid_intervals) < 2:
        return 0, 0

    # 使用多种统计方法
    median_interval = np.median(valid_intervals)
    mean_interval = np.mean(valid_intervals)

    # 计算间隔的一致性
    consistency = 1.0 / (1.0 + np.std(valid_intervals) / np.mean(valid_intervals))

    # 选择更一致的估计
    if consistency > 0.7:  # 高一致性时使用均值
        final_interval = mean_interval
    else:  # 低一致性时使用中位数
        final_interval = median_interval

    bpm = round(60 / final_interval)
    confidence = consistency * min(1.0, len(valid_intervals) / 5.0)  # 置信度

    return bpm, confidence

'''
def detect_peaks_relaxed(x, num_train=8, num_guard=2, rate_fa=1e-1):
    # 放宽的CA-CFAR峰值检测算法，使用更宽松的参数以提高检测敏感度

    num_cells = len(x)
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half

    # 降低alpha值以提高敏感度
    alpha = 0.05 * num_train * (rate_fa ** (-1 / num_train) - 1)

    peak_idx = []
    for i in range(num_side, num_cells - num_side):
        if i != i - num_side + np.argmax(x[i - num_side: i + num_side + 1]):
            continue

        sum1 = np.sum(x[i - num_side: i + num_side + 1])
        sum2 = np.sum(x[i - num_guard_half: i + num_guard_half + 1])
        p_noise = (sum1 - sum2) / num_train
        threshold = alpha * p_noise

        # 降低阈值要求
        if x[i] > threshold * 0.5 and x[i] > np.mean(x) * 0.1:
            peak_idx.append(i)

    return np.array(peak_idx, dtype=int)
'''

def dft_amp(signal):
    # 求离散傅里叶变换的幅值
    dft = fft(signal)
    dft = np.abs(dft)
    return dft


def respiration_freq_amp_ratio(dft_s, st_ix, ed_ix):
    # 计算呼吸频率范围内的频率幅值之和,与全部频率幅值之和的比值
    return np.sum(dft_s[st_ix:ed_ix]) / (np.sum(dft_s) + 1e-6)


class ESP32RespirationAnalyzer:
    # ESP32呼吸检测分析器

    def __init__(self, csv_file_path, sampling_rate=20):
        self.original_fs = 100
        self.csv_file_path = csv_file_path
        self.fs = sampling_rate
        self.sample_gap = int(self.original_fs / self.fs)

        # CSI数据
        self.csi = None
        self.csi_amplitude = None
        self.csi_phase = None

        # 处理后的数据
        self.csi_amplitude_filtered = None
        self.csi_phase_filtered = None
        self.csi_amplitude_emd = None
        self.csi_phase_emd = None
        self.csi_dft_amp = None
        self.csi_respiration_freq_ratio = None

        # 分析结果
        self.respiration_results = {}

    def read_esp32_data(self):
        # 读取ESP32 CSV数据
        print(f'Reading ESP32 CSV data: {self.csv_file_path}')

        try:
            df = pd.read_csv(self.csv_file_path)
            print(f'CSV file shape: {df.shape}')

            # 处理CSI数据：从CSI_DATA列中提取为复数数组
            csi_list = []
            for idx, row in enumerate(df['data'][::self.sample_gap]):
                if idx % 100 == 0:
                    print(f'Process the {idx}th row...')
                cleaned = str(row).replace('[', '').replace(']', '').replace(',', '').strip()

                try:
                    int_vals = [int(x) for x in cleaned.split() if x.strip()]

                    # 确保为偶数个数值
                    usable_len = (len(int_vals) // 2) * 2

                    if usable_len >= 104:
                        int_vals = int_vals[:usable_len]

                        complex_arr = np.array(int_vals, dtype=np.float32)
                        real = complex_arr[::2]
                        imag = complex_arr[1::2]
                        complex_csi = real + 1j * imag
                        csi_list.append(complex_csi)
                    else:
                        continue

                except Exception as e:
                    if idx < 10:
                        print(f"Parsing the {idx}th row failed: {e}")
                    continue

            if len(csi_list) == 0:
                print("No successful parsing of any CSI data was achieved.")
                return False

            self.csi = np.array(csi_list)[:, :, np.newaxis, np.newaxis]

            print(f'Successfully read {self.csi.shape[0]} CSI packets, with each packet containing {self.csi.shape[1]} subcarriers.')

            # 检查数据质量
            amp_range = [np.min(np.abs(self.csi)), np.max(np.abs(self.csi))]
            print(f'CSI amplitude range: [{amp_range[0]:.2f}, {amp_range[1]:.2f}]')

            return True

        except Exception as e:
            print(f'Failed to read data from ESP32: {e}')
            return False

    def process_csi_signals(self):
        # 处理CSI信号
        print('Start processing the CSI signal...')

        # 计算振幅和相位
        self.csi_amplitude = np.abs(self.csi)
        self.csi_phase = np.unwrap(np.angle(self.csi), axis=1)

        # 为ESP32的52个子载波创建正确的索引
        # ESP32使用52个子载波，索引从-26到25（包括0）
        esp32_subcarrier_indices = np.arange(-26, 26)
        self.csi_phase = calib(self.csi_phase, esp32_subcarrier_indices)  # 校准相位的值

        # self.csi_phase = np.angle(self.csi)

        print(f'Amplitude data shape: {self.csi_amplitude.shape}')
        print(f'Phase data shape: {self.csi_phase.shape}')

        # 1. 去除静态分量（去均值）
        print('Remove the static component...')
        self.csi_amplitude = self.csi_amplitude - np.mean(self.csi_amplitude, axis=0, keepdims=True)
        self.csi_phase = self.csi_phase - np.mean(self.csi_phase, axis=0, keepdims=True)

        # 2. 中值滤波
        print('Apply median filtering...')
        self.csi_amplitude_filtered = signal.medfilt(self.csi_amplitude, kernel_size=(3, 1, 1, 1))
        self.csi_phase_filtered = signal.medfilt(self.csi_phase, kernel_size=(3, 1, 1, 1))

        # 3. 扩大带通滤波范围，保留更多频率信息
        print('Application of extended band-pass filtering...')
        nyq = 0.5 * self.fs
        lowcut = 0.05
        highcut = 1.2
        low = lowcut / nyq
        high = highcut / nyq

        try:
            # 直接使用 scipy 的 butter 和 filtfilt，不使用 def bandpass_filter
            b, a = signal.butter(2, [low, high], btype='band')

            self.csi_amplitude_filtered = signal.filtfilt(b, a, self.csi_amplitude_filtered, axis=0)
            self.csi_phase_filtered = signal.filtfilt(b, a, self.csi_phase_filtered, axis=0)

        except Exception as e:
            print(f'Band-pass filtering failed. Using low-pass filtering instead: {e}')

            cutoff = 0.67
            norm_cutoff = cutoff / nyq
            b_lp, a_lp = signal.butter(4, norm_cutoff, btype='low')

            self.csi_amplitude_filtered = signal.filtfilt(b_lp, a_lp, self.csi_amplitude_filtered, axis=0)
            self.csi_phase_filtered = signal.filtfilt(b_lp, a_lp, self.csi_phase_filtered, axis=0)


        # 4. 非常保守的EMD处理
        print('Apply the EMD decomposition...')

        amp_std = np.std(self.csi_amplitude_filtered)
        phase_std = np.std(self.csi_phase_filtered)

        # 只在信号非常嘈杂时才使用EMD
        if amp_std > 0.1:
            print("The amplitude signal has a high level of noise. EMD is applied for noise reduction.")
            self.csi_amplitude_emd = np.apply_along_axis(emd_and_rebuild, 0, self.csi_amplitude_filtered.copy())
        else:
            print("The amplitude signal quality is good. Skip EMD.")
            self.csi_amplitude_emd = self.csi_amplitude_filtered.copy()

        if phase_std > 0.1:
            print("The phase signal has a high level of noise. EMD is applied for noise reduction.")
            self.csi_phase_emd = np.apply_along_axis(emd_and_rebuild, 0, self.csi_phase_filtered.copy())
        else:
            print("The phase signal quality is good. Skip EMD.")
            self.csi_phase_emd = self.csi_phase_filtered.copy()

        # 5. 计算FFT频谱
        print('Calculate the FFT spectrum...')
        self.csi_dft_amp = np.apply_along_axis(dft_amp, 0, self.csi_amplitude_emd.copy())

        # 6. 计算呼吸频率占比 - 呼吸频率范围为0.15Hz-0.6Hz
        n = self.csi_dft_amp.shape[0]
        l_ix = int(0.15 * n / self.fs)   # 0.15Hz
        u_ix = int(0.6 * n / self.fs)  # 0.6Hz

        self.csi_respiration_freq_ratio = np.apply_along_axis(
            respiration_freq_amp_ratio, 0, self.csi_dft_amp.copy(), l_ix, u_ix
        )

        print('CSI signal processing completed.')

    def _representative_rate(self, estimates, confidences=None):
        """从一组估计值中计算代表性呼吸率（BPM）"""
        if estimates is None or len(estimates) == 0:
            return 0
        estimates_array = np.array(estimates)
        if confidences is not None and len(confidences) == len(estimates):
            w = np.array(confidences)
            w = w / (np.sum(w) + 1e-6)
            # 加权中位数
            sorted_idx = np.argsort(estimates_array)
            cumsum_w = np.cumsum(w[sorted_idx])
            med_idx = np.searchsorted(cumsum_w, 0.5)
            rep = estimates_array[sorted_idx][med_idx]
        else:
            rep = np.median(estimates_array)
        return int(round(rep))

    def detect_respiration_rate(self):
        # 检测呼吸速率
        print('Start detecting the respiratory rate...')

        all_estimates = []
        subcarrier_results = []
        method_results = {'fft': [], 'peaks': [], 'confidence': [], 'fft_conf': [], 'peaks_conf': []}

        n_packets, n_subcarriers, n_rx, n_tx = self.csi.shape

        for i in range(n_rx):
            for j in range(n_tx):
                # 计算每个子载波的信号质量指标
                variances = np.var(self.csi_amplitude_emd[:, :, i, j], axis=0)
                freq_ratios = self.csi_respiration_freq_ratio[:, i, j]

                # 综合评分：方差 × 频率占比
                quality_scores = variances * freq_ratios

                # 选择质量最高的子载波（最佳的前15个）
                top_indices = np.argsort(quality_scores)[-min(15, n_subcarriers):]

                for k in top_indices:
                    amplitude_signal = self.csi_amplitude_emd[:, k, i, j]
                    phase_signal = self.csi_phase_emd[:, k, i, j]

                    # 检查信号有效性
                    if np.std(amplitude_signal) < 0.005:
                        continue

                    # 方法1：增强FFT分析
                    amp_bpm_fft, amp_power = enhanced_fft_analysis(amplitude_signal, self.fs)
                    phase_bpm_fft, phase_power = enhanced_fft_analysis(phase_signal, self.fs)

                    # 方法2：改进的峰值检测
                    amp_peaks = advanced_peak_detection(amplitude_signal, self.fs)
                    phase_peaks = advanced_peak_detection(phase_signal, self.fs)

                    amp_bpm_peaks, amp_confidence = improved_peak_rate_calculation(amp_peaks, self.fs)
                    phase_bpm_peaks, phase_confidence = improved_peak_rate_calculation(phase_peaks, self.fs)

                    # 收集所有有效估计
                    estimates = []
                    confidences = []

                    if 5 <= amp_bpm_fft <= 30:
                        estimates.append(amp_bpm_fft)
                        confidences.append(amp_power)
                        method_results['fft'].append(amp_bpm_fft)
                        method_results['fft_conf'].append(amp_power)

                    if 5 <= phase_bpm_fft <= 30:
                        estimates.append(phase_bpm_fft)
                        confidences.append(phase_power)
                        method_results['fft'].append(phase_bpm_fft)
                        method_results['fft_conf'].append(phase_power)

                    if 5 <= amp_bpm_peaks <= 30:
                        estimates.append(amp_bpm_peaks)
                        confidences.append(amp_confidence)
                        method_results['peaks'].append(amp_bpm_peaks)
                        method_results['peaks_conf'].append(amp_confidence)

                    if 5 <= phase_bpm_peaks <= 30:
                        estimates.append(phase_bpm_peaks)
                        confidences.append(phase_confidence)
                        method_results['peaks'].append(phase_bpm_peaks)
                        method_results['peaks_conf'].append(phase_confidence)

                    if estimates:
                        result = {
                            'subcarrier': k,
                            'rx': i,
                            'tx': j,
                            'estimates': estimates,
                            'confidences': confidences,
                            'quality_score': quality_scores[k],
                            'signal_std': np.std(amplitude_signal),
                            'freq_ratio': freq_ratios[k]
                        }
                        subcarrier_results.append(result)
                        all_estimates.extend(estimates)
                        method_results['confidence'].extend(confidences)

        print(f'Found {len(subcarrier_results)} valid subcarriers')
        print(f'Total of {len(all_estimates)} BPM estimates collected')

        if len(all_estimates) == 0:
            print('No valid breathing signal was detected.')
            final_bpm = 0
            detection_status = 'No breathing was detected.'
            weighted_mean = 0
            weighted_median = 0
            cluster_mean = 0
        else:
            # 高级统计融合
            estimates_array = np.array(all_estimates)
            confidences_array = np.array(method_results['confidence'])

            # 按置信度加权的统计
            if len(confidences_array) > 0:
                # 归一化置信度
                confidences_norm = confidences_array / (np.sum(confidences_array) + 1e-6)

                # 加权平均
                weighted_mean = np.average(estimates_array, weights=confidences_norm)

                # 加权中位数
                sorted_indices = np.argsort(estimates_array)
                cumsum_weights = np.cumsum(confidences_norm[sorted_indices])
                median_idx = np.searchsorted(cumsum_weights, 0.5)
                weighted_median = estimates_array[sorted_indices[median_idx]]
            else:
                weighted_mean = np.mean(estimates_array)
                weighted_median = np.median(estimates_array)

            # 使用聚类分析找出主要的估计值
            from scipy.cluster.hierarchy import fcluster, linkage
            if len(estimates_array) > 3:
                # 执行层次聚类
                linkage_matrix = linkage(estimates_array.reshape(-1, 1), method='ward')
                clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')

                # 找到最大的聚类
                unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
                main_cluster = unique_clusters[np.argmax(cluster_counts)]

                # 主聚类的统计值
                main_cluster_estimates = estimates_array[clusters == main_cluster]
                cluster_mean = np.mean(main_cluster_estimates)
                cluster_std = np.std(main_cluster_estimates)
            else:
                cluster_mean = weighted_mean
                cluster_std = np.std(estimates_array)

            print(f'Statistical Analysis:')
            print(f'Weighted Mean: {weighted_mean:.1f} BPM')
            print(f'Weighted Median: {weighted_median:.1f} BPM')
            print(f'Main Cluster Mean: {cluster_mean:.1f} BPM')

            # 最终决策：综合多种统计方法
            if cluster_std < 3.0:  # 聚类一致性好
                final_bpm = int(round(cluster_mean))
            elif abs(weighted_mean - weighted_median) < 2.0:  # 加权统计一致性好
                final_bpm = int(round(weighted_mean))
            else:  # 使用加权中位数（最鲁棒）
                final_bpm = int(round(weighted_median))

            detection_status = f'Breathing detected: {final_bpm} breaths/min'
            print(f'Final Result: {detection_status}')

        # 分别给出FFT和Peak方法的代表性呼吸率
        fft_bpm = self._representative_rate(method_results.get('fft', []), method_results.get('fft_conf', []))
        peak_bpm = self._representative_rate(method_results.get('peaks', []), method_results.get('peaks_conf', []))
        if fft_bpm or peak_bpm:
            print(f"Representative Rates -> FFT: {fft_bpm} BPM, Peaks: {peak_bpm} BPM")

        # 保存详细结果
        self.respiration_results = {
            'detection_status': detection_status,
            'final_bpm': final_bpm,
            'fft_bpm': fft_bpm,          
            'peak_bpm': peak_bpm,         
            'weighted_mean': weighted_mean if len(all_estimates) else 0,
            'weighted_median': weighted_median if len(all_estimates) else 0,
            'cluster_mean': cluster_mean if len(all_estimates) else 0,
            'all_estimates': all_estimates,
            'method_results': method_results,
            'subcarrier_results': subcarrier_results,
            'n_valid_subcarriers': len(subcarrier_results)
        }

        return self.respiration_results

    def visualize_results(self):
        # 可视化分析结果：每一个小图单独生成一个 figure
        if self.csi is None:
            print('Please handle the data first.')
            return

        time_axis = np.arange(self.csi_amplitude.shape[0]) / self.fs

        # 选择最佳子载波
        if self.respiration_results.get('subcarrier_results'):
            best_result = max(self.respiration_results['subcarrier_results'],
                              key=lambda x: x['quality_score'])
            selected_subcarrier = best_result['subcarrier']
            print(f'Selected the best subcarrier {selected_subcarrier} for visualization')
        else:
            variances = np.var(self.csi_amplitude_emd[:, :, 0, 0], axis=0)
            selected_subcarrier = np.argmax(variances)
            print(f'Selected the subcarrier with the highest variance {selected_subcarrier} for visualization')

        # -------- 1. 原始CSI振幅 --------
        plt.figure(figsize=(6, 4))
        plt.plot(time_axis, self.csi_amplitude[:, selected_subcarrier, 0, 0], 'b-', linewidth=1)
        plt.title('Raw CSI Amplitude')
        plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True, alpha=0.3)
        plt.savefig('fig1_raw_amplitude.png', dpi=150, bbox_inches='tight')

        # -------- 2. 中值滤波CSI振幅 --------
        plt.figure(figsize=(6, 4))
        plt.plot(time_axis, self.csi_amplitude_filtered[:, selected_subcarrier, 0, 0], 'g-', linewidth=1)
        plt.title('CSI Amplitude After Median Filtering')
        plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True, alpha=0.3)
        plt.savefig('fig2_median_amplitude.png', dpi=150, bbox_inches='tight')

        # -------- 3. EMD处理CSI振幅 --------
        plt.figure(figsize=(6, 4))
        plt.plot(time_axis, self.csi_amplitude_emd[:, selected_subcarrier, 0, 0], 'r-', linewidth=1)
        plt.title('CSI Amplitude After EMD Processing')
        plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.grid(True, alpha=0.3)
        plt.savefig('fig3_emd_amplitude.png', dpi=150, bbox_inches='tight')

        # -------- 4. 原始CSI相位 --------
        plt.figure(figsize=(6, 4))
        plt.plot(time_axis, self.csi_phase[:, selected_subcarrier, 0, 0], 'b-', linewidth=1)
        plt.title('Raw CSI Phase')
        plt.xlabel('Time (s)'); plt.ylabel('Phase (radians)'); plt.grid(True, alpha=0.3)
        plt.savefig('fig4_raw_phase.png', dpi=150, bbox_inches='tight')

        # -------- 5. 中值滤波CSI相位 --------
        plt.figure(figsize=(6, 4))
        plt.plot(time_axis, self.csi_phase_filtered[:, selected_subcarrier, 0, 0], 'g-', linewidth=1)
        plt.title('CSI Phase After Median Filtering')
        plt.xlabel('Time (s)'); plt.ylabel('Phase (radians)'); plt.grid(True, alpha=0.3)
        plt.savefig('fig5_median_phase.png', dpi=150, bbox_inches='tight')

        # -------- 6. EMD处理CSI相位 --------
        plt.figure(figsize=(6, 4))
        plt.plot(time_axis, self.csi_phase_emd[:, selected_subcarrier, 0, 0], 'r-', linewidth=1)
        plt.title('CSI Phase After EMD Processing')
        plt.xlabel('Time (s)'); plt.ylabel('Phase (radians)'); plt.grid(True, alpha=0.3)
        plt.savefig('fig6_emd_phase.png', dpi=150, bbox_inches='tight')

        # -------- 7. 呼吸率估计直方图 --------
        if self.respiration_results.get('all_estimates'):
            plt.figure(figsize=(6, 4))
            estimates = self.respiration_results['all_estimates']
            plt.hist(estimates, bins=min(15, len(estimates)), alpha=0.7,
                     edgecolor='black', color='skyblue')
            plt.axvline(self.respiration_results['final_bpm'], color='red', linestyle='--', linewidth=2,
                        label=f'Final: {self.respiration_results["final_bpm"]}')
            plt.axvline(self.respiration_results['weighted_mean'], color='green', linestyle='--',
                        label=f'Weighted Mean: {self.respiration_results["weighted_mean"]:.1f}')
            plt.axvline(self.respiration_results['cluster_mean'], color='blue', linestyle='--',
                        label=f'Cluster Mean: {self.respiration_results["cluster_mean"]:.1f}')
            plt.title('Distribution of Respiration Rate Estimates')
            plt.xlabel('BPM'); plt.ylabel('Frequency'); plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
            plt.savefig('fig7_histogram.png', dpi=150, bbox_inches='tight')

        # -------- 8. 文字统计 --------
        if self.respiration_results:
            plt.figure(figsize=(6, 4))
            lines = [
                f"Final Respiration Rate: {self.respiration_results.get('final_bpm', 0)} BPM",
                f"FFT Representative: {self.respiration_results.get('fft_bpm', 0)} BPM",
                f"Peak Representative: {self.respiration_results.get('peak_bpm', 0)} BPM",
                f"Weighted Mean: {self.respiration_results.get('weighted_mean', 0):.1f} BPM",
                f"Cluster Mean: {self.respiration_results.get('cluster_mean', 0):.1f} BPM"
            ]
            for i, t in enumerate(lines):
                plt.text(0.05, 0.95 - i*0.09, t, transform=plt.gca().transAxes, fontsize=10, va='top')
            plt.axis('off'); plt.title('Detection Statistics')
            plt.savefig('fig8_statistics.png', dpi=150, bbox_inches='tight')

        plt.show()

    def run_full_analysis(self):
        # 运行完整分析
        print("Starting ESP32 respiration detection analysis...")

        if not self.read_esp32_data():
            return None

        self.process_csi_signals()
        results = self.detect_respiration_rate()
        self.visualize_results()

        return results


def analyze_esp32_respiration(csv_file_path, sampling_rate=20):
    # ESP32呼吸检测主函数
    analyzer = ESP32RespirationAnalyzer(csv_file_path, sampling_rate)
    return analyzer.run_full_analysis()


if __name__ == "__main__":
    print("ESP32 breath detection algorithm")
    print("=" * 50)

    test_file = "csi_data.csv"

    if Path(test_file).exists():
        print(f"Analysis document: {test_file}")
        results = analyze_esp32_respiration(test_file)

        if results:
            print(f"\nAnalysis completed: {results['detection_status']}")
            print(f"FFT Representative Rate: {results.get('fft_bpm', 0)} BPM")
            print(f"Peak Representative Rate: {results.get('peak_bpm', 0)} BPM")
        else:
            print("Analysis failed")
    else:
        print(f"The file does not exist: {test_file}")