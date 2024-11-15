import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv_Blocks(nn.Module):
    def __init__(self, in_channel, out_channel, downsamle=False, upsample=False):
        super(Conv_Blocks, self).__init__()
        self.upsample = upsample
        if not downsamle:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, x):
        x = self.layer(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = x*2
        return x

class Encoder_Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder_Decoder, self).__init__()
        self.in_conv = Conv_Blocks(in_channel, 64, downsamle=False, upsample=False)
        self.down1 = Conv_Blocks(64, 64, downsamle=True, upsample=False)
        self.down2 = Conv_Blocks(64, 64, downsamle=True, upsample=False)
        #self.down3 = Conv_Blocks(64, 64, downsamle=True, upsample=False)
        self.down_up = Conv_Blocks(64, 64, downsamle=True, upsample=True)
        self.up1 = Conv_Blocks(64, 64, downsamle=False, upsample=True)
        self.up2 = Conv_Blocks(64, 64, downsamle=False, upsample=True)
        #self.up3 = Conv_Blocks(64, 64, downsamle=False, upsample=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.out_mask_offset = nn.Sequential(
            nn.Conv2d(64, out_channel, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

    def forward(self, x):
        f1 = self.in_conv(x)
        f2 = self.down1(f1)  #64
        f3 = self.down2(f2)  #32
        #f4 = self.down3(f3)
        f4 = self.down_up(f3) #32
        f5 = self.up1(f3 + f4) #64
        f6 = self.up2(f5 + f2)
        #f8 = self.up3(f7 + f2)
        f = self.out_conv(f6 + f1)
        return self.out_mask_offset(f)
