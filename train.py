import random
from tqdm import tqdm
import torch
import glob
import argparse
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import numpy as np
import os
import logging
import sys
from model.network import BaseLine_CNN_follow_JBHI2024
from dataloader.pcg import PhysioNetDataset
from utils.training_utils import get_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.evaluate import Evaluate_get_metrics

parser = argparse.ArgumentParser()


# ==============lr===================
parser.add_argument('--base_lr',type=float,default=0.0001)
parser.add_argument('--lr_decay_patience',type=int,default=6)
# ==============lr===================

# ==============training params===================
parser.add_argument('--fold',type=int,default=1)
parser.add_argument('--num_classes',type=int,default=2)
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--val_batch_size',type=int,default=4)
parser.add_argument('--max_epoch',type=int,default=1000)
parser.add_argument('--early_stop_patience',type=int,default=30)
parser.add_argument('--save_period',type=int,default=10)

parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--num_works',type=int,default=0)

parser.add_argument('--val_period',type=int,default=1)
parser.add_argument('--exp',type=str,default='PhysioNetCinC_Challenge_2016')
parser.add_argument('--dataset_name',type=str,default='PhysioNetCinC_Challenge_2016')
parser.add_argument('--optim',type=str,default='Adam')

parser.add_argument('--autodl',action='store_true')


def create_version_folder(snapshot_path):
    # 检查是否存在版本号文件夹
    version_folders = [name for name in os.listdir(snapshot_path) if name.startswith('version')]

    if not version_folders:
        # 如果不存在版本号文件夹，则创建version0
        new_folder = os.path.join(snapshot_path, 'version0')
    else:
        # 如果存在版本号文件夹，则创建下一个版本号文件夹
        last_version = max(version_folders)
        last_version_number = int(last_version.replace('version', ''))
        next_version = 'version{:02d}'.format(last_version_number + 1)
        new_folder = os.path.join(snapshot_path, next_version)

    os.makedirs(new_folder)
    return new_folder


args = parser.parse_args()
snapshot_path = "./exp_2d_pcg/" + args.exp + "/"
base_lr = args.base_lr


if __name__ == '__main__':


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    snapshot_path = create_version_folder(snapshot_path)

    device = "cuda:{}".format(args.device)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # init model
    model = BaseLine_CNN_follow_JBHI2024(in_channels=1,num_classes=args.num_classes)
    model.to(device)

    # init dataset
    root_base = 'E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/training'
    train_csv_path = "E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/annotations/annotations/all_labels_samples.csv"
    val_csv_path = "E:/Deep_Learning_DATABASE/PCG/PhysioNetCinC_Challenge_2016/annotations/annotations/all_labels_samples.csv"
    if args.autodl:
        root_base = '/root/autodl-tmp/PhysioNetCinC_Challenge_2016/training'
        train_csv_path = "./datasets/PhysioNetCinC_Challenge_2016/{}-fold-training.csv".format(args.fold)
        val_csv_path = "./datasets/PhysioNetCinC_Challenge_2016/{}-fold-val.csv".format(args.fold)

    train_dataset = PhysioNetDataset(root_base,train_csv_path)

    train_dataloder = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_works,
                                pin_memory=True,
                                drop_last= True,
                                shuffle=True
                                    )

    val_dataset = PhysioNetDataset(root_base,val_csv_path,training=False)

    val_dataloder = DataLoader(val_dataset,batch_size=args.val_batch_size,num_workers=1,drop_last=False)

    val_iteriter = tqdm(val_dataloder)
    model.train()
    # init optimizer
    optimizer = get_optimizer(model=model,name=args.optim,base_lr=args.base_lr)

    writer = SummaryWriter(snapshot_path + '/log')
    if args.num_classes == 2:
        criteria = BCEWithLogitsLoss()
    else:
        criteria = CrossEntropyLoss()

    evaluator = Evaluate_get_metrics(device=device)

    iter_num = 0
    max_epoch = args.max_epoch
    lr_ = args.base_lr
    best_f1 = 0.0
    no_improve_epoch = 0
    # 初始化学习率调整策略
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_decay_patience, verbose=True)

    model.train()
    print("=================共计训练epoch: {}====================".format(max_epoch))
    # 开始训练
    iterator = tqdm(range(max_epoch))
    for epoch_num in iterator:
        torch.cuda.empty_cache()
        time1 = time.time()
        for i_batch,batch_data in enumerate(train_dataloder):
            time2 = time.time()

            input_data, labels = batch_data
            input_data, labels = input_data.to(device), labels.to(device)

            outputs = model(input_data).squeeze()

            loss = criteria(outputs[:,1,...],labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)



        # eval
        with torch.no_grad():
                model.eval()
                show_id = random.randint(0,len(val_iteriter))
                for id_val,val_data in enumerate(val_iteriter):
                    input_data, labels = val_data
                    input_data, labels = input_data.to(device), labels.to(device)
                    outputs = model(input_data)
                    if args.num_classes == 2:
                        outputs = torch.sigmoid(outputs)
                        preds = (outputs[:,1,:] > 0.5).float()
                    else:
                        preds = torch.softmax(outputs,dim=1).argmax().squeeze()
                    evaluator.add(preds,labels.unsqueeze(1))
                metrics = evaluator.compute()
                Sensitivity = metrics['Sensitivity']
                Specificity = metrics['Specificity']
                Precision = metrics['Precision']
                Accuracy = metrics['Accuracy']
                F1 = metrics['F1-score']

                evaluator.reset()
                writer.add_scalar('val/Sensitivity', Sensitivity, epoch_num)
                writer.add_scalar('val/Specificity', Specificity, epoch_num)
                writer.add_scalar('val/Precision', Precision, epoch_num)
                writer.add_scalar('val/Accuracy', Accuracy, epoch_num)
                writer.add_scalar('val/F1-score', F1, epoch_num)

                # 检查F1指标是否有提升
                if F1 > best_f1:
                    best_f1 = F1
                    no_improve_epoch = 0
                    name = "best_f1" + str(round(best_f1, 4)) + '_epoch_' + str(epoch_num) + '.pth'
                    save_mode_path = os.path.join(
                        snapshot_path, name)
                    previous_files = glob.glob(os.path.join(snapshot_path, '*best_f1*.pth'))
                    for file_path in previous_files:
                        os.remove(file_path)

                    torch.save(model.state_dict(), save_mode_path)

                    print("Epoch {}: New best F1-score: {:.4f}. Model saved.".format(epoch_num, best_f1))
                else:
                    no_improve_epoch += 1
                    print("Epoch {}: No improvement in F1-score. Best is {:.4f}.".format(epoch_num, best_f1))

                # 如果F1指标没有提高，则通过scheduler降低学习率
                scheduler.step(F1)

                # 如果F1指标连续多个epoch没有提高，提前终止训练
                if no_improve_epoch >= args.early_stop_patience:
                    print("Early stopping triggered after {} epochs without improvement in F1-score.".format(
                        no_improve_epoch))
                    break

                if epoch_num % args.save_period == 0:
                    name = 'epoch_' + str(epoch_num) + '.pth'
                    save_mode_path = os.path.join(
                        snapshot_path, name)
                    torch.save(model.state_dict(), save_mode_path)



    writer.close()
