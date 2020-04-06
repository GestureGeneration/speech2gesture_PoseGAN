#!/usr/bin/python
# -*- coding: utf-8 -*-

# import torch module
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim

# import python module
import time
import os
import argparse
import datetime
from pathlib import Path
from distutils.util import strtobool

# import modules
import numpy as np

from model import get_model
from loss import cal_g_loss
from loss import cal_d_loss
from log_output import write_parameters
from log_output import save_history

def get_argument():
    """
    実験設定の取得

    Returns
    -------
    args: Namespace
        コマンドライン引数から作成した実験設定
    """
    parser = argparse.ArgumentParser(description='text to gesture by PyTorch')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default:256)')
    parser.add_argument('--epochs', type=int, default=200, help='number of the epoch to train (default:200)')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for training (default:0.001)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (default:0)')
    # parser.add_argument('--embedding_dimension', type=int, default=300, help='dimension of embedded feature (default:300)')
    parser.add_argument('--embedding_dimension', type=int, default=26, help='dimension of mfcc feature (default:26)')
    parser.add_argument('--outdir_path', type=str, default='./out/', help='directory path of outputs(default:./out)')
    parser.add_argument('--gpu_num', type=int, default='0', help='gpu number(default:0)')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    # parser.add_argument('--txt_path', type=str, default='./data/X_train_posegan3.npy', help='text path')
    parser.add_argument('--speech_path', type=str, default='./X_train_posegan.npy', help='text path')
    parser.add_argument('--pose_path', type=str, default='./Y_train_posegan_norm.npy', help='pose path')
    parser.add_argument('--generator', type=str, default='unet_decoder', help='choose generator name')
    parser.add_argument('--gan', type=strtobool, default=1, help='GAN usage(default:1)')
    parser.add_argument('--discriminator', type=str, default='patchgan', help='choose discriminator name')
    parser.add_argument('--lambda_d', type=float, default='1.', help='lambda(default:1)')
    args = parser.parse_args()
    return args

def main(args):
    """
    学習と評価を行うメイン関数

    Returns
    -------
    net: type(model)
        最終エポックの学習モデル
    loss_history: dict
        TrainとValidationのLoss推移データ
    """
    # Load the dataset
    # text
    # textのshapeは(data_size, embedding_dimension, frames)
    txt = np.load(args.speech_path)

    # pose
    # poseのshapeは(data_size, joint_points, frames)
    pose = np.load(args.pose_path)

    # Prepare DataLoaders
    # 読み込んだデータセットからDataLoaderの準備
    train_data_size = int(len(txt)*0.8)
    tr_tensor_dataset = data_utils.TensorDataset(torch.tensor(txt[:train_data_size], dtype=torch.float),
                                                 torch.tensor(pose[:train_data_size], dtype=torch.float))
    val_tensor_dataset = data_utils.TensorDataset(torch.tensor(txt[train_data_size:], dtype=torch.float),
                                                  torch.tensor(pose[train_data_size:], dtype=torch.float))
    train_loader = data_utils.DataLoader(tr_tensor_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = data_utils.DataLoader(val_tensor_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    loaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': train_data_size, 'val': len(txt) - train_data_size}
    
    print('Complete the preparing dataset')

    # Set the gpu usage
    # argsのdeviceを確認してcpuかgpuを設定
    device = torch.device('cuda:' + str(args.gpu_num) if args.device == 'cuda' else 'cpu')
    print('device: ', device)

    # Set the network
    # argsで指定されたモデルのネットワーク定義
    # Generator
    g_model = get_model(args.generator)
    in_channels = args.embedding_dimension
    out_channels = 256  # PozeGANに合わせる
    g_net = g_model(in_channels, out_channels, args.batch_size, device)
    g_net.to(device)
    g_optim = optim.Adam(g_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Discriminator
    if args.gan:
        d_model = get_model(args.discriminator)
        d_net = d_model()
        d_net.to(device)
        d_optim = optim.Adam(d_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define a Loss function and optimizer
    # 損失関数の定義
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    # initialize the loss and accuracy history
    # 学習推移データ格納用変数の定義
    loss_history = {'g_train': [], 'g_val': [], 'd_train': [], 'd_val': []}

    # Train the network
    start_time = time.time()
    for epoch in range(args.epochs):
        print('* ' * 20)
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('* ' * 20)

        # Each epoch has a training and validation phase
        # TrainとValidationを交互に実行
        for phase in ['train', 'val']:
            if phase == 'train':
                g_net.train(True)
                if args.gan:
                    d_net.train(True)
            else:
                g_net.train(False)
                if args.gan:
                    d_net.train(False)

            # initialize the running loss and corrects
            g_running_loss = 0.0
            d_running_loss = 0.0

            # ミニバッチをDataLoaderから取り出す
            for i, (inputs, corrects) in enumerate(loaders[phase]):
                inputs.requires_grad, corrects.requires_grad = True, True

                # データにdeviceを設定
                inputs, corrects = inputs.to(device), corrects.to(device)

                # zero the parameter gradients
                g_optim.zero_grad()
                if args.gan:
                    d_optim.zero_grad()

                # forward
                if phase == 'train':
                    # Generator Lossの算出
                    g_loss, g_outputs = cal_g_loss(inputs, corrects, g_net, l1_criterion)
                    loss = g_loss
                    if args.gan:
                        # Discriminator Lossの算出
                        d_loss = cal_d_loss(g_outputs, corrects, d_net, mse_criterion, args.lambda_d, device)
                        loss = loss + d_loss
                    # Lossの逆伝播
                    loss.backward()
                    g_optim.step()
                    if args.gan:
                        d_optim.step()
                else:
                    # Validation時は計算グラフを保存しない処理を行う
                    with torch.no_grad():
                        # Generator Lossの算出
                        g_loss, g_outputs = cal_g_loss(inputs, corrects, g_net, l1_criterion)
                        if args.gan:
                            # Discriminator Lossの算出
                            d_loss = cal_d_loss(g_outputs, corrects, d_net, mse_criterion, args.lambda_d, device)

                # バッチ毎のLossを加算
                g_running_loss += g_loss.item()
                if args.gan:
                    d_running_loss += d_loss.item()

            # epoch lossを算出し，学習推移データとして保存
            g_epoch_loss = g_running_loss / dataset_sizes[phase]
            loss_history['g_' + phase].append(g_epoch_loss)
            print('{} Generator Loss: {:.20f}'.format(phase, g_epoch_loss))

            if args.gan:
                d_epoch_loss = d_running_loss / dataset_sizes[phase]
                loss_history['d_' + phase].append(d_epoch_loss)
                print('{} Discriminator Loss: {:.20f}'.format(phase, d_epoch_loss))

    # Trainingにかかった時間を算出・表示
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    learned_net = [g_net]
    if args.gan:
        learned_net.append(d_net)

    # 学習済みモデルと学習推移データを返す
    return learned_net, loss_history


if __name__ == '__main__':
    # get the time stamp
    # 実行日時の取得
    time_stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # get the arguments and write the log
    # コマンドライン引数（実験設定）の取得
    args = get_argument()

    # make directory to save outcomes

    # もしgpuが使用できない場合は，実験設定をcpuで上書き
    if not torch.cuda.is_available():
        args.device = 'cpu'

    # train the network and output the result
    # モデルを学習し，一番低いValidation Lossを出したネットワークと，学習の推移データを受け取る
    nets, loss_history = main(args)
    
    # 結果保存用フォルダの作成と，実験設定の書き出し
    outdir_path = args.outdir_path + time_stamp + '/'
    os.makedirs(outdir_path, exist_ok=True)
    write_parameters(args, outdir_path, time_stamp)

    # 学習済みネットワークを保存
    for name, net in zip(['generator', 'discriminator'], nets):
        torch.save(net.state_dict(), Path(outdir_path).joinpath('{}_{}_weights.pth'.format(name, time_stamp)))

    # 学習の推移データを保存
    save_history(loss_history, outdir_path, args.epochs, time_stamp)

