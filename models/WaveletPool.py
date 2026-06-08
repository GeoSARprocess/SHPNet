import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveletPool(nn.Module):
    """
    小波池化层 - 使用Haar小波进行频域分解下采样
    
    特点：
    1. 零参数：使用固定的Haar小波滤波器
    2. 信息无损：通过频域分解保留100%信息
    3. 通道扩展：输出通道数为输入的4倍
    4. 空间下采样：高宽各降低2倍
    """
    
    def __init__(self):
        super(WaveletPool, self).__init__( )
        
        # 定义Haar小波的4个滤波器
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])      # 低频分量
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])    # 垂直高频
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])    # 水平高频
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])    # 对角高频
        
        # 堆叠滤波器并翻转（卷积需要）
        filts = np.stack([
            ll[None, ::-1, ::-1],
            lh[None, ::-1, ::-1],
            hl[None, ::-1, ::-1],
            hh[None, ::-1, ::-1]
        ], axis=0)
        
        # 注册为不可训练参数
        self.register_buffer(
            'weight',
            torch.tensor(filts, dtype=torch.float32)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图 (B, C, H, W)
        
        Returns:
            y: 输出特征图 (B, 4C, H/2, W/2)
               包含LL, LH, HL, HH四个频域分量
        """
        B, C, H, W = x.shape
        
        # 为每个输入通道复制滤波器
        filters = self.weight.repeat(C, 1, 1, 1)  # (4C, 1, 2, 2)
        
        # 分组卷积实现小波变换
        y = F.conv2d(x, filters, stride=2, groups=C)
        
        return y
    
    def get_frequency_components(self, x):
        """
        获取分离的频域分量（用于可视化和分析）
        
        Returns:
            ll, lh, hl, hh: 四个频域分量
        """
        y = self.forward(x)
        C = x.shape[1]
        
        # 分离四个分量
        ll = y[:, 0::4, :, :]  # 低频
        lh = y[:, 1::4, :, :]  # 垂直高频
        hl = y[:, 2::4, :, :]  # 水平高频
        hh = y[:, 3::4, :, :]  # 对角高频
        
        return ll, lh, hl, hh


class WaveletDownBlock(nn.Module):
    """
    带通道调整的小波下采样块
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.wavelet = WaveletPool()
        # 4倍通道扩展后降维到目标通道数
        self.channel_adjust = nn.Conv2d(
            in_channels * 4, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.wavelet(x)
        x = self.channel_adjust(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 使用示例
if __name__ == '__main__':
    # 创建测试输入
    x = torch.randn(2, 64, 32, 32)
    
    # 基础WaveletPool
    wavelet = WaveletPool()
    y = wavelet(x)
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")  # (2, 256, 16, 16)
    
    # 获取频域分量
    ll, lh, hl, hh = wavelet.get_frequency_components(x)
    print(f"LL(低频): {ll.shape}")
    print(f"LH(垂直): {lh.shape}")
    print(f"HL(水平): {hl.shape}")
    print(f"HH(对角): {hh.shape}")
    
    # 带通道调整的下采样块
    down_block = WaveletDownBlock(64, 64)
    y2 = down_block(x)
    print(f"下采样块输出: {y2.shape}")  # (2, 128, 16, 16)

