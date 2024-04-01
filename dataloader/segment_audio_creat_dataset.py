from torch.utils.data import Dataset
import os
from scipy import signal
import samplerate
import numpy as np
import librosa

def standard_normal_variate(data):
    data = data / np.max(np.abs(data))
    return data

def band_pass_filter(original_signal, order, fc1, fc2, fs):
    '''
    中值滤波器
    :param original_signal: 音频数据
    :param order: 滤波器阶数
    :param fc1: 截止频率
    :param fc2: 截止频率
    :param fs: 音频采样率
    :return: 滤波后的音频数据
    '''
    b, a = signal.butter(N=order, Wn=[2 * fc1 / fs, 2 * fc2 / fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal

def down_sample(audio_data):
    data = samplerate.resample(audio_data.T, 1000 / fs, converter_type='sinc_best').T
    return data




if __name__ == '__main__':
    csv_path = "E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/annotations/annotations/all_lables.csv"
    root = 'E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/training'
    with open(csv_path,'r') as f:
        all_data_info = f.read().splitlines()
    for i in all_data_info:
        database_name = i.split(',')[0]
        current_name = i.split(',')[1]
        current_org_label = i.split(',')[2]
        current_label = 0 if current_org_label == -1 else 1

        current_data_folder = os.path.join(root,database_name)
        out_data_folder = os.path.join(root,database_name)
        out_data_folder = out_data_folder.replace('2016/training','2016/h5_files')
        if not os.path.exists(out_data_folder):
            os.makedirs(out_data_folder)


        current_data_path = os.path.join(current_data_folder,current_name+'.wav')
        audio_data, fs = librosa.load(current_data_path, sr=None)
        # 中值滤波
        audio_data = band_pass_filter(audio_data, 2, 25, 400, fs)
        # 下采样
        audio_data = samplerate.resample(audio_data.T, 1000 / fs, converter_type='sinc_best').T
        # 归一化
        norm_data = standard_normal_variate(audio_data)

        # 根据标签进行segment
        window_length = 5000
        sliding_step_size = 2500 if current_label == 0  else 650
        total_num = len(norm_data) / (window_length)

        print(len(audio_data))
        print(total_num)
        break
