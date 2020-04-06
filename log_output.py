#!/usr/bin/python
# -*- coding: utf-8 -*-

# import modules
from pathlib import Path
import csv
import numpy as np
import os
import matplotlib.pyplot as plt

def save_history(loss_history, outdir, epochs, time_stamp):
    """
    実験結果を保存する

    Parameters
    ----------
    loss_history: dict
        学習推移データ
    outdir: string
        出力パス
    epochs: int
        学習エポック数
    time_stamp: string
        実験開始時のタイムスタンプ
    """

    training_history = np.zeros((4, epochs))
    for j, phase in enumerate(['g_train', 'g_val', 'd_train', 'd_val']):
        if len(loss_history[phase]) == 0:
            break
        training_history[j] = loss_history[phase]
    np.save(Path(outdir).joinpath('{}_training_history.npy'.format(time_stamp)), training_history)

    save_lossfig(training_history[:2], ['g_train', 'g_val'], 'Generator Loss', os.path.join(outdir, 'g_loss.png'))
    if len(training_history) > 2:
        save_lossfig(training_history[2:], ['d_train', 'd_val'], 'Discriminator Loss', os.path.join(outdir, 'd_loss.png'))
        # save_lossfig(training_history[::2], ['g_train', 'd_train'], 'G-D Loss', os.path.join(outdir, 'g_vs_d(train).png'))
        # save_lossfig(training_history[1::2], ['d_val', 'd_val'], 'G-D Loss', os.path.join(outdir, 'g_vs_d(val).png'))

def save_lossfig(train_his, label, title, file_name):
    """
    学習曲線グラフのpng出力

    Parameters
    ----------
    train_his: numpy.ndarray
        学習推移データ
    label: list
        グラフラベル名
    title: string
        グラフタイトル
    file_name: string
        グラフのファイル名
    """
    epochs = np.arange(len(train_his[0])) + 1
    plt.title(title)
    plt.plot(epochs, train_his[0], label=label[0])
    plt.plot(epochs, train_his[1], label=label[1])
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def write_parameters(args, outdir_path, time_stamp):
    """
    実験設定をcsvに書き出す

    Parameters
    ----------
    args: Namespace
        実験設定
    outdir_path: string
        出力パス
    time_stamp: string
        実行日時
    """

    fout = open(Path(outdir_path).joinpath('{}_experimental_settings.csv'.format(time_stamp)), "wt")
    csvout = csv.writer(fout)
    print('*' * 50)
    print('Parameters')
    print('*' * 50)
    for arg in dir(args):
        if not arg.startswith('_'):
            csvout.writerow([arg,  str(getattr(args, arg))])
            print('%-25s %-25s' % (arg, str(getattr(args, arg))))


