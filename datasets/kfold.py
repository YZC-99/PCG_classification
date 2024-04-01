

"""
我需要进行10-fold交叉验证，因此要根据all_labels_path文件进行10次数据切分
all_labels_path文件内容预览如下：
training-a,a0001,1,1
training-a,a0002,1,1
training-a,a0003,1,1
training-a,a0004,1,1
training-a,a0005,1,1
training-a,a0006,1,0
第一列是data_base_name，第二列是data_name,第三列是label，-1是正常，1是异常

数据划分后的需求
1、根据data_base_name和label进行分层抽样，也就是data_base_name和data_name的分布都要尽量和原来保持一致
2、划分后的数据保存为和原来csv格式一样的形式，并保存在out_path，并命令为k-fold-training.csv和k-fold-val.csv
"""
# all_labels_path = './PhysioNetCinC_Challenge_2016/all_labels.csv'
# out_path = './PhysioNetCinC_Challenge_2016'
# ==========================
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# import os
#
# # 读取CSV文件
# all_labels_path = './PhysioNetCinC_Challenge_2016/all_labels.csv'
# out_path = './PhysioNetCinC_Challenge_2016'
# df = pd.read_csv(all_labels_path, header=None, names=['data_base_name', 'data_name', 'label', 'unused_label'])
#
# # 创建一个新列，结合`data_base_name`和`label`作为分层的依据
# df['stratify_group'] = df['data_base_name'] + "_" + df['label'].astype(str)
#
# # 初始化StratifiedKFold对象
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
# # 开始10-fold划分
# for fold, (train_index, val_index) in enumerate(skf.split(df, df['stratify_group']), 1):
#     train_df, val_df = df.iloc[train_index], df.iloc[val_index]
#
#     # 在保存之前去除`stratify_group`列
#     train_df = train_df.drop(columns=['stratify_group'])
#     val_df = val_df.drop(columns=['stratify_group'])
#
#     # 保存划分后的数据
#     train_df.to_csv(os.path.join(out_path, f'{fold}-fold-training.csv'), index=False, header=False)
#     val_df.to_csv(os.path.join(out_path, f'{fold}-fold-val.csv'), index=False, header=False)
#
#     print(f'Fold {fold}: Training and validation datasets are saved.')

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os


def log_distribution(df, title, file):
    total_samples = len(df)
    message = f"{title}:\n"
    message += f"Total samples: {total_samples}\n"
    message += "Distribution of data_base_name:\n"
    message += df['data_base_name'].value_counts(normalize=True).to_string() + "\n"
    message += "Distribution of label:\n"
    message += df['label'].value_counts(normalize=True).to_string() + "\n"
    message += "Distribution of label within each data_base_name:\n"
    for name, group in df.groupby('data_base_name'):
        message += f"{name}:\n"
        message += group['label'].value_counts(normalize=True).to_string() + "\n"
    message += "\n" + "-" * 50 + "\n"

    print(message)
    file.write(message + "\n")


# 读取CSV文件
all_labels_path = './PhysioNetCinC_Challenge_2016/all_labels.csv'
out_path = './PhysioNetCinC_Challenge_2016'
distribution_file_path = os.path.join(out_path, 'data_distribution.txt')
df = pd.read_csv(all_labels_path, header=None, names=['data_base_name', 'data_name', 'label', 'unused_label'])

# 创建一个新列，结合`data_base_name`和`label`作为分层的依据
df['stratify_group'] = df['data_base_name'] + "_" + df['label'].astype(str)

with open(distribution_file_path, 'w') as file:
    # 打印原始数据的分布
    log_distribution(df, "Original Data Distribution", file)

    # 初始化StratifiedKFold对象
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 开始10-fold划分
    for fold, (train_index, val_index) in enumerate(skf.split(df, df['stratify_group']), 1):
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]

        # 打印每个fold的训练和验证集的分布
        log_distribution(train_df, f"Fold {fold} Training Data Distribution", file)
        log_distribution(val_df, f"Fold {fold} Validation Data Distribution", file)

        # 在保存之前去除`stratify_group`列
        train_df = train_df.drop(columns=['stratify_group'])
        val_df = val_df.drop(columns=['stratify_group'])

        # 保存划分后的数据
        train_df.to_csv(os.path.join(out_path, f'{fold}-fold-training.csv'), index=False, header=False)
        val_df.to_csv(os.path.join(out_path, f'{fold}-fold-val.csv'), index=False, header=False)
