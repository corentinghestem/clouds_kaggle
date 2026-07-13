import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinConfig, SwinModel

from src.config import Config


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SwinUNet(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            self.encoder = SwinModel.from_pretrained(backbone_name, add_pooling_layer=False)
        else:
            self.encoder = SwinModel(SwinConfig.from_pretrained(backbone_name), add_pooling_layer=False)

        dim = self.encoder.config.embed_dim  # 96 pour swin-tiny

        self.up1 = DecoderBlock(dim * 8, dim * 4, 256)
        self.up2 = DecoderBlock(256, dim * 2, 128)
        self.up3 = DecoderBlock(128, dim, 64)
        self.up4 = DecoderBlock(64, 0, 32)
        self.up5 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, pixel_values):
        out = self.encoder(pixel_values, output_hidden_states=True)
        f0, f1, f2, _, f4 = out.reshaped_hidden_states

        x = self.up1(f4, f2)
        x = self.up2(x, f1)
        x = self.up3(x, f0)
        x = self.up4(x)
        x = self.up5(x)
        return self.head(x)


def build_model() -> SwinUNet:
    model = SwinUNet(
        backbone_name=Config.BACKBONE_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED_BACKBONE,
    )
    return model.to(Config.DEVICE)
