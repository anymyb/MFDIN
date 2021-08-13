''' network architecture for MFDIN '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
try:
    from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class D_Align(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(D_Align, self).__init__()
        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        offset = torch.cat([nbr_fea_l, ref_fea_l], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        f_fea = self.lrelu(self.dcnpack([nbr_fea_l, offset]))
        return f_fea

class MFDIN_OLD1P(nn.Module):
    def __init__(self, nf=64, groups=8, front_RBs=5, back_RFAs=2, center=None, nfields=5):
        super(MFDIN_OLD1P, self).__init__()
        self.nf = nf
        self.center = nfields // 2 if center is None else center
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        RFABlock = functools.partial(arch_util.RFABlock, nf=nf)
        #### extract features (for each frame)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)  #前残差组提取特征||test分组卷积
        self.fea_upconv = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)
        self.fea_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.AlignMoudle = D_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(nfields * nf, nf, 1, 1, bias=True)  #普通融合卷积函数
        #### reconstruction
        self.recon_trunk = arch_util.make_layer(RFABlock, back_RFAs)  #后残差组重建图片
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def V_shuffle(self, inputs, scale):
        N, C, iH, iW = inputs.size()
        oH = iH * scale
        oW = iW
        oC = C // scale
        output = inputs.view(N, oC, scale, iH, iW)
        output = output.permute(0, 1, 3, 2, 4).contiguous()
        output = output.view(N, oC, oH, oW)
        return output

    def forward(self, x):
        B, N, C, H, W = x.size()  
        fields = []
        fields.append(x[:, 0, :, :, :][:, :, ::2, :])
        fields.append(x[:, 0, :, :, :][:, :, 1::2, :])
        fields.append(x[:, 1, :, :, :][:, :, ::2, :])
        fields.append(x[:, 1, :, :, :][:, :, 1::2, :])
        fields.append(x[:, 2, :, :, :][:, :, ::2, :])
        fields = torch.stack(fields, dim=1)
        N = 5
        #### extract LR features
        fea1 = self.lrelu(self.conv_first(fields.view(-1, C, H//2, W)))
        f_fea = self.feature_extraction(fea1)
        f_fea = self.lrelu(self.fea_upconv(f_fea+fea1))
        f_fea = self.lrelu(self.fea_conv1(self.V_shuffle(f_fea, 2)))
        f_fea = f_fea.view(B, N, -1, H, W)  #[4, 6, 64, 64, 64]
        #### pcd align
        # ref feature list
        ref_fea_lA = f_fea[:, self.center, :, :, :].clone()
        aligned_feaA = []
        for i in range(N):
            nbr_fea_lA = f_fea[:, i, :, :, :].clone()
            aligned_feaA.append(self.AlignMoudle(nbr_fea_lA, ref_fea_lA))
        aligned_feaA = torch.stack(aligned_feaA, dim=1).view(B, -1, H, W)
        fea = self.fusion(aligned_feaA)
        out = self.recon_trunk(fea)
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)

        return out
