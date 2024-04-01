import librosa
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，可以替换为你需要的后端

from scipy import signal

import samplerate

def plot_signal(audio_data, title=None):
    plt.figure(figsize=(12, 3.5), dpi=300)
    plt.plot(audio_data, linewidth=1)
    plt.title(title,fontsize = 16)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.show()

"""
由于音频在制作时不可避免地会保存一部分噪声，我们需对音频文件进行数字滤波，旨在滤除高频噪声以及直流噪声，同时尽可能保留心音信号。
我们把音频送入二阶25-400hz的巴特沃斯中值滤波器，并可视化音频。实现代码如下：
"""
def band_pass_filter(original_signal, order, fc1,fc2, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fc2: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal



if __name__ == '__main__':
    audio_path = "E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/training/training-d/d0001.wav"
    audio_data, fs = librosa.load(audio_path, sr=None)

    # plot_signal(audio_data, title='Initial Audio')
    #
    # audio_data = band_pass_filter(audio_data, 2, 25, 400, fs)
    # plot_signal(audio_data, title='After Filter')

    # 下采样
    down_sample_audio_data = samplerate.resample(audio_data.T, 1000 / fs, converter_type='sinc_best').T
    plot_signal(down_sample_audio_data, title='Down_sampled')

    # 归一化
    down_sample_audio_data = down_sample_audio_data / np.max(np.abs(down_sample_audio_data))
    plot_signal(down_sample_audio_data, title='Normalized')

    total_num = len(down_sample_audio_data) / (5000)  # 计算切割次数
    fig = plt.figure(figsize=(12, 5), dpi=300)

    ax1 = fig.add_subplot(2, 1, 1)
    plt.plot(down_sample_audio_data, linewidth=1)
    plt.title('Cut Audio(With Overlap)', fontsize=16)
    plt.tick_params(labelsize=12)
    plt.ylim([-1.2, 1.2])
    plt.grid()
    for j in range(int(total_num)):
        plt.vlines(j * 5000, -1.2, 1.2, color="red", linestyle='--', linewidth=1.1)

    ax2 = fig.add_subplot(2, 1, 2)
    plt.plot(down_sample_audio_data, linewidth=1)
    for j in range(int(total_num)):
        plt.vlines(j * 5000 + 2500, -1.2, 1.2, color="green", linestyle='--', linewidth=1.1)
    plt.ylim([-1.2, 1.2])
    plt.grid()
    plt.show()
