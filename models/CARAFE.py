import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv


class CARAFE(nn.Module):
    """
    CARAFE 是一种上采样模块，通过学习的权重对特征图进行上采样。
    参数:
        c (int): 输入通道数
        k_enc (int): 编码器部分的卷积核大小
        k_up (int): 上采样时使用的 unfold 核大小
        c_mid (int): 中间通道数
        scale (int): 上采样倍率
    """

    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super(CARAFE, self).__init__()
        self.scale = scale  # 设置上采样倍率

        # 压缩输入通道到中间通道数
        self.comp = Conv(c, c_mid)

        # 编码器生成权重，输出通道数为(scale * k_up)^2
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)

        # 使用 PixelShuffle 进行上采样操作
        self.pix_shf = nn.PixelShuffle(scale)

        # 最近邻插值方法作为上采样操作
        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')

        # Unfold 操作提取感受野内的特征
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()  # 获取输入张量的形状
        h_, w_ = h * self.scale, w * self.scale  # 计算上采样后的高度和宽度

        W = self.comp(X)  # 压缩输入通道
        W = self.enc(W)  # 编码器生成权重
        W = self.pix_shf(W)  # Pixel Shuffle 上采样权重
        W = torch.softmax(W, dim=1)  # 对权重应用 softmax 归一化

        X = self.upsmp(X)  # 输入特征图使用最近邻插值上采样
        X = self.unfold(X)  # 提取上采样后特征的感受野
        X = X.view(b, c, -1, h_, w_)  # 调整视图以分离感受野维度

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # 应用注意力机制加权聚合
        return X  # 返回上采样后的特征图



# 使用示例
if __name__ == '__main__':
    # 创建测试输入
    x = torch.randn(2, 64, 32, 32)
    up_block = CARAFE(64)
    y2 = up_block(x)
    print(f"采样块输出: {y2.shape}")  #