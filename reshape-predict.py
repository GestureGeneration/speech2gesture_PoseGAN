"""
predictをreshape

pose => pose_reshaped
(block,192,64) => (frames,192)

example
(2,192,64) => (128,192)
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
from distutils.util import strtobool
import glob

parser = argparse.ArgumentParser(description='speech to gesture by PyTorch')
parser.add_argument('--denorm', '-d', type=strtobool, default=1, help='denorm(1) or not denorm(0)')
parser.add_argument('--denormpath', type=str, default="./norm/", help='denorm path')
parser.add_argument('--datatype', type=str, default="train",help='denorm datatype(train or dev)')
parser.add_argument('--npypath', type=str, default="./test_inputs/*20200404-182816_weights.npy",help='npyfiles path')
parser.add_argument('--outpath', type=str, default="./predict_reshaped/" ,help='out path')
args = parser.parse_args()

def main():
    files = glob.glob(args.npypath)
    print("処理ファイル数",len(files))
    os.makedirs(args.outpath, exist_ok=True)

    for filename in files:        
        print("process file...",filename)
        if os.path.exists(filename):
            predict = np.load(filename)
            pose_seq = np.array([])
            for i,data in enumerate(predict):
                data_transpose = np.transpose(data)
                if i == 0:
                    pose_seq = data_transpose
                else:
                    pose_seq = np.append(pose_seq,data_transpose,axis=0)
            
            # 学習データを標準化していたならば、元に戻す(denorm)
            if args.denorm:
                pose_seq = data_denorm(pose_seq)
            
            filename2 = (filename.split("/")[-1]).split("-")[0]

            # save
            if args.denorm:
                np.savetxt(args.outpath+"{}_posegan-denorm.txt".format(filename2),pose_seq)
            else:
                np.savetxt(args.outpath+"{}_posegan.txt".format(filename2),pose_seq)

def data_denorm(pose_seq):
    print("-denorm-")
    ave = np.load(args.denormpath + "ave_"+args.datatype+"_posegan.npy")
    std = np.load(args.denormpath + "std_"+args.datatype+"_posegan.npy")
    pose_seq_denorm = pose_seq*std+ave
    return pose_seq_denorm

if __name__ == '__main__':
    main()

    print("--complete--")
