#!/usr/bin/python
# -*- coding: utf-8 -*-

# import modules
import torch


def diff_frames(poses):
    """
    前後フレーム間差分ベクトルを算出

    Parameters
    ----------
    poses: torch.Tensor
        ポーズのミニバッチ
        shapeは(batch_size, joint_points, frames)

    Returns
    -------
    torch.Tensor
        前後フレーム間差分ベクトルを並べた行列
    """
    pre_poses = poses[:, :, :-1]
    next_poses = poses[:, :, 1::]

    return next_poses - pre_poses

def cal_g_loss(inputs, corrects, g_net, criterion):
    """
    Generator Lossの算出
    Parameters
    ----------
    inputs: torch.Tensor
        入力のミニバッチ
    corrects: torch.Tensor
        正解データのミニバッチ
    g_net:
        Generatorのネットワーク
    criterion:
        Loss関数（PoseGANではL1）

    Returns
    -------
    g_loss: torch.Tensor
        算出したLoss
    g_outputs: torch.Tensor
        Generatorが生成したデータ
    """
    # Generator Lossの算出
    g_outputs = g_net(inputs)
    pose_loss = criterion(g_outputs, corrects)  # 各座標点のMSE Loss(PoseGANのL1 Loss)
    motion_loss = criterion(diff_frames(g_outputs), diff_frames(corrects))  # 次フレームと差分ベクトルのL1 Loss
    g_loss = pose_loss + motion_loss

    return g_loss, g_outputs

def cal_d_loss(g_outputs, corrects, d_net, criterion, lambda_d, device):
    """
    Generator Lossの算出
    Parameters
    ----------
    g_outputs: torch.Tensor
        Generatorの出力
    corrects: torch.Tensor
        正解データのミニバッチ
    d_net:
        Discriminatorのネットワーク
    criterion:
        Loss関数（PoseGANではMSE）
    lambda_d: float
        loss計算に使用するハイパーパラメータ（PoseGANでは1.00）

    Returns
    -------
    d_loss: torch.Tensor
        算出したLoss
    """
    # Discriminator Lossの算出
    d_real_outputs = d_net(corrects)
    d_fake_outputs = d_net(g_outputs)
    d_loss = criterion(torch.ones(d_real_outputs.shape).to(device), d_real_outputs) + lambda_d\
        * criterion(torch.zeros(d_fake_outputs.shape).to(device), d_fake_outputs)

    return d_loss
