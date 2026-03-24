import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torchsummary import summary



"""
初始化网络层的权重参数
CycleGAN 论文推荐使用均值为 0、标准差为 0.02 的正态分布初始化卷积层
实例归一化层（InstanceNorm2d）权重固定为 1，偏置为 0，符合论文设定
偏置项统一初始化为 0，避免额外噪声

Args:
    net (nn.Module): 需要初始化的网络（Generator/Discriminator）
    init_type (str): 初始化类型，可选 'normal' | 'xavier' | 'kaiming' | 'orthogonal'
    init_gain (float): 初始化分布的增益/标准差

example:
    # 1. 初始化生成器
    gen = Generator(in_ch=3, out_ch=3).to(device)
    gen = init_weights(gen, init_type='normal', init_gain=0.02)  # CycleGAN 论文推荐 normal 初始化
    print("生成器参数初始化完成")
"""
def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        # 对卷积层、转置卷积层初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'初始化方法 {init_type} 未实现')

            # 偏置项初始化为 0（如果有）
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # 实例归一化层的权重初始化为 1，偏置为 0
        elif classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    # 递归初始化所有子模块
    net.apply(init_func)
    return net



# ====================== 2. 生成器 (对应论文 4. 网络架构) ======================
# 论文：9个残差块 + 下采样 + 上采样 + 实例归一化
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        # 初始卷积层
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ] # 64x256x256

        # 下采样 (论文：2层步长为2的卷积)
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        # 256x64x64

        # 残差块 (论文：256×256图像使用9个残差块)
        for _ in range(9):
            model += [ResidualBlock(in_features)]
        # 256x64x64

        # 上采样 (论文：2层分数步长卷积)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # 输出层
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_ch, 7),
            nn.Tanh()  # 输出 [-1,1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)




# ====================== 3. 判别器 (论文 4. PatchGAN 70×70) ======================
'''
传统 GAN 判别器输出单个数值（判断整张图真假），容易忽略局部瑕疵；而 PatchGAN 设计为输出 2D 特征图，让判别器逐局部评估真假，能更精准地约束生成图像的细节质量。
判别器的有效感受野大小为 70×70（即输出特征图的每个像素，对应输入图像中 70×70 的局部块）；
256×256 输入经过多步下采样后，最终输出特征图的尺寸自然为 30×30（计算逻辑：(256 - 70) / 1 + 1 = 187？不，实际是通过卷积层逐步缩小：256→128→64→32→30，核心是感受野覆盖 70×70 时，输出尺寸固定为 30×30）。
'''
class Discriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        model = [
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":

    # 以下内容是用于测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # G = Generator(3,3).to(device)
    #
    # t_in = torch.rand((2,3,256,256)).to(device)
    # t_out = G(t_in)
    # print(t_in.shape,t_out.shape)
    # exit()
    #
    # # ==================== 关键：输入是 3×256×256 ====================
    # print("=" * 50)
    # print("        CycleGAN 生成器模型结构 (输入：3通道 256×256)")
    # print("=" * 50)
    # summary(G, (3, 256, 256))  # (通道, 高, 宽)

    D = Discriminator(3).to(device)
    t_in = torch.rand((2,3,256,256)).to(device)
    t_out = D(t_in)
    print(t_in.shape,t_out.shape)
    exit()