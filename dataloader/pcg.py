import torch
from torch.utils.data import Dataset
import os
from scipy import signal
# import samplerate
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

def down_sample(audio_data,fs):
    data = librosa.resample(audio_data, orig_sr=fs, target_sr=1000)
    return data

class PhysioNetDataset(Dataset):
    def __init__(self,root,csv_path,training=True):

        self.root = root
        self.training = training
        self.window_length= 5000
        self.segments = []
        tag = 'training' if training else 'val'
        with open(csv_path, 'r') as f:
            self.all_data_info = f.read().splitlines()
        print("================={}共{}条心音样本===============".format(tag,len(self.all_data_info)))
        for data_info in self.all_data_info:
            database_name = data_info.split(',')[0]
            current_name = data_info.split(',')[1]
            current_org_label = data_info.split(',')[2]
            current_label = 0 if current_org_label == '-1' else 1
            if training:
                sliding_step_size = 2500 if current_org_label == '-1' else 650
            else:
                sliding_step_size = 2500

            current_data_path = os.path.join(self.root,database_name, current_name + '.wav')
            audio_data, fs = librosa.load(current_data_path, sr=None)
            # 中值滤波
            audio_data = band_pass_filter(audio_data, 2, 25, 400, fs)
            # 下采样
            audio_data = down_sample(audio_data,fs)
            # 归一化
            norm_data = standard_normal_variate(audio_data)

            for start in range(0, len(norm_data), sliding_step_size):
                segment = norm_data[start:start + self.window_length]
                # 如果segment的长度小于window_length并且是最后一次循环，则取末尾的window_length长度的数据
                if len(segment) < self.window_length:
                    segment = norm_data[-self.window_length:]
                self.segments.append((segment, current_label))
                # 如果已经取了末尾的window_length长度的数据，则终止循环
                if len(segment) < self.window_length:
                    break
        print("================={}共{}个样本===============".format(tag,len(self.segments)))
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, item):
        segment, label = self.segments[item]
        segment = torch.from_numpy(segment).float()
        segment = torch.unsqueeze(segment,dim=0)
        label = torch.tensor(int(label)).long()
        return segment, label

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    csv_path = "E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/annotations/annotations/all_labels_samples.csv"
    root = 'E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/training'
    dataset1 = PhysioNetDataset(root=root,csv_path=csv_path)
    dataloader = DataLoader(dataset1,batch_size=2)
    for audio_data,label in dataloader:
        print(audio_data.shape)
        print(label.shape)
        break
