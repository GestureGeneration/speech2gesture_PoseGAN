"""
audio  => pose
(block,26,64) => (block,192,64)
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
# import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def main():
    parser = argparse.ArgumentParser(description='speech to gesture by PyTorch')
    parser.add_argument('--modelpath', type=str, default="./out/20200404-182816/generator_20200404-182816_weights.pth", help='model path') # Please fix modelpath
    parser.add_argument('--inputpath', type=str, default="./test_inputs/", help='input path')
    parser.add_argument('--outpath', type=str, default="./predict/", help='out path')
    args = parser.parse_args()

    os.makedirs(args.outpath, exist_ok=True)

    GPU = False
    device = torch.device("cuda" if GPU else "cpu")
    # g_net = g_model(26, 256, args.batch_size, device)
    g_model = get_model("unet_decoder")
    model = g_model(26, 256, 256, device)
    model.eval()
    # args.path = './model/train9e200n/generator_20191212-183801_weights.pth'
    model.load_state_dict(torch.load(args.modelpath,map_location=device))

    for num in range(1093,1183):
        PATH = args.ipnutpath + "X_test_gesture"+str(num)+"_posegan.npy"
        if os.path.exists(PATH):
            testnpdata = np.load(PATH)
            testdata = torch.tensor(testnpdata, dtype=torch.float).to(device)
            predict_tensor = model(testdata)
            print(predict_tensor.shape)
            predict_np = predict_tensor.detach().numpy()
            filename = (args.modelpath.split("/")[-1])

            np.save(args.outpath+"/gesture"+str(num)+"-{}".format(filename.replace(".pth",".npy")),predict_np)

if __name__ == '__main__':
    main()
    print("--complete--")
