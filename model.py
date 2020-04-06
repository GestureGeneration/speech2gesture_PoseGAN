#!/usr/bin/python
# -*- coding: utf-8 -*-

# import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormRelu2d(nn.Module):
    """(conv => BN => ReLU)"""
    def __init__(self, in_channels, out_channels, k, s, p=1):
        super(ConvNormRelu2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
       
    def forward(self, x):
        x = self.conv(x)
        return x

class ConvNormRelu1d(nn.Module):
    """(conv => BN => ReLU)"""
    def __init__(self, in_channels, out_channels, k, s, p=1):
        super(ConvNormRelu1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
       
    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv1d(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, k, s, p=1):
        super(DoubleConv1d, self).__init__()
        self.block = nn.Sequential(
            ConvNormRelu1d(in_channels, out_channels, k, s, p),
            ConvNormRelu1d(out_channels, out_channels, k, s, p),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class DoubleConv2d(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, k, s, p=1):
        super(DoubleConv2d, self).__init__()
        self.block = nn.Sequential(
            ConvNormRelu2d(in_channels, out_channels, k, s, p),
            ConvNormRelu2d(out_channels, out_channels, k, s, p),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Down1d(nn.Module):
    """Max Pooling => Double Conv"""
    def __init__(self, in_channels, out_channels):
        super(Down1d, self).__init__()
        self.block = nn.Sequential(
            # nn.MaxPool1d(2, 2),
            ConvNormRelu1d(in_channels, out_channels, k=4, s=2)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Up1d(nn.Module):
    """Up sampling??? => add => Double Conv"""
    def __init__(self, in_channels, out_channels):
        super(Up1d, self).__init__()
        self.block = nn.Sequential(
            DoubleConv1d(in_channels, out_channels, k=3, s=1)
        )

    def forward(self, x, y):
        """PoseGANの実装そのまま"""
        x = torch.repeat_interleave(x, 2, dim=2)
        x = x + y
        x = self.block(x)
        return x

class UNet1d(nn.Module):
    """
    Text Encoder
    """
    def __init__(self, in_channels):
        super(UNet1d, self).__init__()
        self.inconv = DoubleConv1d(in_channels, 256, k=3, s=1)
        self.down1 = Down1d(256, 256)
        self.down2 = Down1d(256, 256)
        self.down3 = Down1d(256, 256)
        self.down4 = Down1d(256, 256)
        self.down5 = Down1d(256, 256)
        self.up1 = Up1d(256, 256)
        self.up2 = Up1d(256, 256)
        self.up3 = Up1d(256, 256)
        self.up4 = Up1d(256, 256)
        self.up5 = Up1d(256, 256)
        self.up6 = Up1d(256, 256)
        
    def forward(self, x):
        x0 = self.inconv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)
        return x

class Decoder(nn.Module):
    """
    CNN Decoder
    """
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            DoubleConv1d(in_channels, out_channels, k=3, s=1),
            DoubleConv1d(out_channels, out_channels, k=3, s=1),
            DoubleConv1d(out_channels, out_channels, k=3, s=1),
            DoubleConv1d(out_channels, out_channels, k=3, s=1),
            nn.Conv1d(out_channels, 192, kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x


class PatchGan(nn.Module):
    """
    Motion Discriminator
        forwardへのinput shapeは(batch_size, 98, 64)を想定
    """

    # def __init__(self, in_channel=98, ndf=64):
    def __init__(self, in_channel=192, ndf=64):
        """
        Parameter
        ----------
        in_channel: int
            入力チャネル数
        ndf: int(default=64)
            Size of feature maps in discriminator
        """
        super(PatchGan, self).__init__()
        self.layer1 = nn.Conv1d(in_channel, ndf, kernel_size=4, stride=2, padding=0)
        self.layer2 = nn.LeakyReLU(0.2, inplace=True)
        self.layer3 = ConvNormRelu1d(ndf, ndf * 2, k=4, s=2, p=1)
        self.layer4 = ConvNormRelu1d(ndf * 2, ndf * 4, k=4, s=1, p=0)
        self.layer5 = nn.Conv1d(ndf * 4, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.layer1(F.pad(x, [1, 2], "constant", 0))
        x = self.layer2(x)
        x = self.layer3(F.pad(x, [1, 2], "constant", 0))
        x = self.layer4(F.pad(x, [1, 2], "constant", 0))
        x = self.layer5(F.pad(x, [1, 2], "constant", 0))

        return x

class EncoderGRU(nn.Module):
    """
    Text Encoder
    """
    def __init__(self, batch_size, device, embedding_dim=64, hidden_dim=300, cnn_dim=64, channels=256):
        super(EncoderGRU, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, channels*cnn_dim)

        self.initial_hidden_state = torch.zeros(2, batch_size, hidden_dim, device=device)
        self.batch_size = batch_size
        self.cnn_dim = cnn_dim
        self.channels = channels

    def forward(self, x):
        _, x = self.gru(x, self.initial_hidden_state)
        x = F.relu(self.fc(torch.cat([x[0], x[1]], dim=1)))

        return x.view([self.batch_size, self.channels, self.cnn_dim])  

class GRU_Unet_Decoder(nn.Module):
    """
    gru => unet => cnn_decoderモデル
    ※ Parameterをモデル間で統一
    """
    def __init__(self, in_channels, out_channels, batch_size, device):
        """
        Parameter
        ----------
        in_channels : int
            入力チャネル数
        out_channels: int
            出力チャネル数
        batch_size: int
            バッチサイズ
        device:
            cpu or gpu の設定
        """
        super(GRU_Unet_Decoder, self).__init__()
        self.gru = EncoderGRU(batch_size, device)
        self.unet = UNet1d(in_channels)
        self.decoder = Decoder(out_channels, out_channels)
        self.batch_size = batch_size
        self.device = device

    def forward(self, x):
        pad_frag = False  # paddingは，入力発話文が可変長のときに実行
        if not len(x) == self.batch_size:
            pad_frag = True
            pad_len = self.batch_size - len(x)
            pad = torch.zeros(pad_len, 300, 64)
            x = torch.cat((x, pad.to(self.device)), dim=0)
        x = self.gru(x)
        x = self.unet(x)
        x = self.decoder(x)
        if pad_frag:
            x = x[:-pad_len]
        return x

class Unet_Decoder(nn.Module):
    """
    unet => cnn_decoderモデル
    ※ Parameterをモデル間で統一
    """
    def __init__(self, in_channels, out_channels, batch_size, device):
        """
        Parameter
        ----------
        in_channels : int
            入力チャネル数
        out_channels: int
            出力チャネル数
        batch_size: int
            バッチサイズ
        device:
            cpu or gpu の設定
        """
        super(Unet_Decoder, self).__init__()
        self.unet = UNet1d(in_channels)
        self.decoder = Decoder(out_channels, out_channels)

    def forward(self, x):
        x = self.unet(x)
        x = self.decoder(x)
        return x


models = {
    'gru_unet_decoder': GRU_Unet_Decoder,
    'unet_decoder': Unet_Decoder,
    'patchgan': PatchGan
}

def get_model(name):
    """
    名前を指定してモデルを取得する

    Parameters
    ----------
    name: string
        モデルの名前

    Returns
    -------
    models[name]: class
        指定したモデルのクラス

    """
    return models[name]
