__all__ = ['PatchWaveNet']

# Cell
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from mamba_ssm import Mamba
from layers.MSGBlock import GraphBlock, Attention_Block

class TemporalExternalAttn(nn.Module):
    def __init__(self, d_model, S=512):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries):

        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S

        out = self.mv(attn)  # bs,n,d_model
        return out


class WaveletTransform1D(nn.Module):
    def __init__(self, configs, wavelet='haar', device='cuda', revin = True, affine = True, subtract_last = False):
        super().__init__()
        self.lookback = configs.seq_len
        self.patch_size = configs.patch_len
        self.d_model = configs.d_model
        # self.d_ff = configs.d_ff
        self.stride = configs.stride
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.block = nn.ModuleList()
        for i in [1, 2, 3]:
            self.block.append(
                ScaleGraphBlock2(configs, self.d_model // (2**i))
            )
        # 初始化小波滤波器
        if wavelet == 'haar':
            self.low_pass_filter = torch.tensor([0.707, 0.707], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.707, 0.707], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        elif wavelet == 'db2':
            self.low_pass_filter = torch.tensor([0.48296, 0.8365, 0.2241, -0.1294], dtype=torch.float32,
                                                device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.1294, -0.2241, 0.8365, -0.48296], dtype=torch.float32,
                                                 device=device).unsqueeze(0).unsqueeze(0)
        elif wavelet == 'db3':
            self.low_pass_filter = torch.tensor([0.3327, 0.8069, 0.4599, -0.1350, -0.0854, 0.0352], dtype=torch.float32,
                                                device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.0352, -0.0854, 0.1350, 0.4599, -0.8069, 0.3327],
                                                 dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        elif wavelet == 'db4':
            self.low_pass_filter = torch.tensor([0.2304, 0.7148, 0.6309, -0.0279, -0.1870, 0.0308, 0.0329, -0.0106],
                                                dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            self.high_pass_filter = torch.tensor([-0.0106, -0.0329, 0.0308, 0.1870, -0.0279, -0.6309, 0.7148, -0.2304],
                                                 dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError("Only 'haar', 'db2', 'db3', and 'db4' wavelets are implemented for this example.")
            # 根据输入通道数调整滤波器大小
        self.low_pass_filter = self.low_pass_filter.expand(self.patch_num, 1, self.low_pass_filter.size(-1))
        self.high_pass_filter = self.high_pass_filter.expand(self.patch_num, 1, self.high_pass_filter.size(-1))

    def forward(self, x):
        coffs = self.multi_level_decompose(x, 3)
        restruct = self.multi_level_reconstruct(coffs)
        return restruct

    def dwt_step(self, x):
        """
        单级小波分解
        :param x: 输入信号，形状 [B, C, T]
        :return: 低频和高频分量
        """
        low_pass = F.conv1d(x, self.low_pass_filter, stride=2, groups=x.size(1), padding=(self.low_pass_filter.size(-1) - 1) // 2)
        high_pass = F.conv1d(x, self.high_pass_filter, stride=2, groups=x.size(1), padding=(self.high_pass_filter.size(-1) - 1) // 2)
        return low_pass, high_pass

    def idwt_step(self, low_pass, high_pass):
        """
        单级小波重构
        :param low_pass: 低频分量
        :param high_pass: 高频分量
        :return: 重构的信号
        """
        low_reconstructed = F.conv_transpose1d(low_pass, self.low_pass_filter, stride=2, groups=low_pass.size(1), padding=(self.low_pass_filter.size(-1) - 1) // 2)
        high_reconstructed = F.conv_transpose1d(high_pass, self.high_pass_filter, stride=2, groups=high_pass.size(1), padding=(self.high_pass_filter.size(-1) - 1) // 2)
        return low_reconstructed + high_reconstructed

    def multi_level_decompose(self, x, level):
        """
        多级小波分解
        :param x: 输入信号，形状 [B, C, T]
        :param level: 分解层数
        :return: 各级分解的低频和高频分量
        """
        coefficients = []
        current_signal = x
        for i in range(level):
            low_pass, high_pass = self.dwt_step(current_signal)
            coefficients.append((low_pass, high_pass))  # 保存每级低频和高频分量
            current_signal = low_pass  # 继续分解低频分量
        return coefficients

    def multi_level_reconstruct(self, coefficients):
        """
        多级小波重构
        :param coefficients: 各级分解的低频和高频分量
        :return: 每一级别的重构信号列表和最终重构信号
        """
        # reconstructed_levels = []  # 保存每一级重构信号
        # low_freq_info = []  # 保存每一级低频信息
        # high_freq_info = []  # 保存每一级高频信息
        i = 2
        current_signal = None

        # 从最高层开始逐级重构
        for low_pass, high_pass in reversed(coefficients):

            if current_signal is None:
                low_pass = self.block[i](low_pass)
                high_pass = self.block[i](high_pass)
                # 初始重构
                current_signal = self.idwt_step(low_pass, high_pass)
            else:
                current_signal = self.block[i](current_signal)
                high_pass = self.block[i](high_pass)
                # 在每一级的基础上加入高频分量进行重构
                current_signal = self.idwt_step(current_signal, high_pass)
            i = i-1
            # # 保存当前级别的重构信号
            # reconstructed_levels.append(current_signal)
            # low_freq_info.append(low_pass)  # 保存低频信息
            # high_freq_info.append(high_pass)  # 保存高频信息

        # 返回重构信号列表、低频和高频信息
        return current_signal


class ScaleGraphBlock2(nn.Module):
    def __init__(self, configs, d_model):
        super(ScaleGraphBlock2, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = d_model
        # self.d_ff = d_ff
        # self.patch_num = patch_num
        self.norm = nn.LayerNorm(self.d_model)
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.gelu = nn.GELU()
        # self.gconv = nn.ModuleList()
        # self.GraphConv = GraphBlock(self.patch_size, self.d_model, configs.conv_channel, configs.skip_channel, configs.gcn_depth, configs.dropout, configs.propalpha, self.patch_num, configs.node_dim)
        # self.mamba = Mamba(self.d_model, d_state=configs.d_state, d_conv=configs.d_conv)
        # self.att0 = Attention_Block(self.d_model, self.d_ff,
        #                          n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.att1 = TemporalExternalAttn(self.d_model, (2*self.d_model))

    def forward(self, x):
        # x = self.GraphConv(x)  # [B, C, d_model]
        # for Mul-attetion
        # out = self.norm(self.att0(x))
        # out = self.norm(x)
        out = self.norm(self.att1(x))
        out = self.gelu(out)
        res = out + x
        # res = self.mamba(res)
        return res

class ScaleGraphBlock1(nn.Module):
    def __init__(self, configs, patch_num, d_model):
        super(ScaleGraphBlock1, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = d_model
        self.patch_num = patch_num
        self.norm = nn.LayerNorm(self.d_model)
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.gelu = nn.GELU()
        # self.gconv = nn.ModuleList()
        self.GraphConv = GraphBlock(self.patch_size, self.d_model, configs.conv_channel, configs.skip_channel, configs.gcn_depth, configs.dropout, configs.propalpha, self.patch_num, configs.node_dim)
        self.mamba = Mamba(self.d_model, d_state=configs.d_state, d_conv=configs.d_conv)
        # self.att0 = Attention_Block(self.d_model, configs.d_ff,
        #                          n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        # self.att1 = TemporalExternalAttn(self.d_model, (2*self.d_model))

    def forward(self, x):
        x = self.GraphConv(x)  # [B, C, d_model]
        # for Mul-attetion
        # out = self.norm(self.att0(x))
        # out = self.norm(x)
        out = self.norm(self.mamba(x))
        out = self.gelu(out)
        res = out + x
        # res = self.mamba(res)
        return res

class Backbone(nn.Module):
    def __init__(self, configs, revin = True, affine = True, subtract_last = False):
        super().__init__()
        self.enc_in = configs.enc_in
        self.lookback = configs.seq_len
        self.pred = configs.pred_len
        self.nvals = configs.enc_in
        self.batch_size = configs.batch_size
        self.patch_size = configs.patch_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.dropout = nn.Dropout(0.2)
        self.stride = configs.stride
        self.fatten = nn.Flatten(start_dim=-2)
        # self.series_decom = series_decomp(25)
        self.norm = nn.LayerNorm(configs.d_model)
        self.patch_num = int((self.lookback - self.patch_size) / self.stride + 1) + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)
        self.gelu = nn.GELU()
        self.mlp1 = nn.Linear(self.patch_size, self.d_model)
        self.mlp2 = nn.Linear(self.d_model*self.patch_num, int(self.pred * 2))
        self.mlp3 = nn.Linear(int(self.pred * 2), self.pred)
        # self.mamba = Mamba(self.d_model, d_state=configs.d_state, d_conv=configs.d_conv)
        self.block = nn.ModuleList()
        for _ in range(configs.e_layers):
            self.block.append(
                ScaleGraphBlock1(configs, self.patch_num, self.d_model)
            )
        self.wavenet = WaveletTransform1D(configs, wavelet='haar', device='cuda')

    def forward(self, x):
        B, T, N = x.size()
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = self.mlp1(x)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.wavenet(x)
        for i in range(self.e_layers):
            mamba = self.block[i](x)
        x_out = self.fatten(mamba)
        x_out = torch.reshape(x_out, (B, N, -1))
        x_out = self.mlp2(x_out)
        x_out = self.gelu(x_out)
        x_out = self.mlp3(x_out)
        x_out = x_out.permute(0, 2, 1)
        if self.revin:
            x_out = self.revin_layer(x_out, 'denorm')
        return x_out

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)

    def forward(self, x):
        x = self.model(x)
        return x
