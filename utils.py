import random
import time
import datetime
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt




'''
1.初始化：创建一个能存最多50张图像的"记忆库"
2.添加新图像：
    如果记忆库没满（少于50张）：直接把新图像存进去
    如果记忆库已满：抛硬币决定（50%概率）
        正面：随机替换记忆库中的一张旧图像
        反面：不保存新图像
3.返回图像：每次返回的图像包含：
    要么是新图像
    要么是记忆库中的旧图像（随机选择）
'''
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)



# ====================== 6. 学习率调度（论文：100轮后线性衰减）======================
'''
n_epochs = 总共训练多少轮
offset = 轮数小偏移（几乎 = 0）
decay_start_epoch = 学习率从第几轮开始下降
1.0  ────────────────
                        \
                         \
                          \
                           \
0.0                           ─────
                  100轮     200轮
'''
class LambdaLR:
    def __init__(self, n_epochs=200, offset=0, decay_start_epoch=100):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)



def tensor_to_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    将经过标准化处理的张量转换回可显示的图像格式

    参数:
        tensor: 输入张量，形状为(C, H, W)
        mean: 标准化时使用的均值，默认为(0.5, 0.5, 0.5)
        std: 标准化时使用的标准差，默认为(0.5, 0.5, 0.5)

    返回:
        numpy数组，形状为(H, W, C)，值在[0, 1]范围内
    """
    # 克隆张量以避免修改原始数据
    tensor = tensor.cpu().detach().clone()
    # 反标准化：将图像从[-1, 1]范围转换回[0, 1]范围[4](@ref)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    # 确保值在[0, 1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    # 将张量从(C, H, W)转换为(H, W, C)格式[6](@ref)
    image = tensor.permute(1, 2, 0)
    # 转换为numpy数组
    image = image.numpy()
    return image


def batch_tensor_to_image(tensor_batch, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    将经过标准化处理的批量张量转换回可显示的图像格式

    参数:
        tensor_batch: 输入张量，形状为(B, C, H, W)
        mean: 标准化时使用的均值，默认为(0.5, 0.5, 0.5)
        std: 标准化时使用的标准差，默认为(0.5, 0.5, 0.5)

    返回:
        numpy数组，形状为(B, H, W, C)，值在[0, 1]范围内
    """
    # 克隆张量并移至CPU
    tensor_batch = tensor_batch.cpu().detach().clone()

    # 将mean和std转换为张量，并调整形状为适合广播的维度 [1, C, 1, 1]
    mean_tensor = torch.tensor(mean).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std).view(1, -1, 1, 1)

    # 批量反标准化：利用广播机制一次性处理整个批次
    tensor_batch = tensor_batch * std_tensor + mean_tensor

    # 确保值在[0, 1]范围内
    tensor_batch = torch.clamp(tensor_batch, 0, 1)

    # 将张量从(B, C, H, W)转换为(B, H, W, C)格式，以适应图像显示库（如Matplotlib）的要求[3,7](@ref)
    image_batch = tensor_batch.permute(0, 2, 3, 1)  # 交换维度顺序

    # 转换为numpy数组
    image_batch = image_batch.numpy()

    return image_batch



def show_fakes(realA,realB,fakeA,fakeB,save=None):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))  # 宽度设为高度的4倍以适应水平排列

    # 显示第一张图片
    axs[0].imshow(realA)
    axs[0].set_title('realA')
    axs[0].axis('off')  # 关闭坐标轴

    # 显示第二张图片
    axs[1].imshow(fakeB)
    axs[1].set_title('fakeB')
    axs[1].axis('off')

    # 显示第三张图片
    axs[2].imshow(realB)
    axs[2].set_title('realB')
    axs[2].axis('off')

    # 显示第四张图片
    axs[3].imshow(fakeA)
    axs[3].set_title('fakeA')
    axs[3].axis('off')

    # 调整子图间距，避免重叠
    plt.tight_layout()
    if save!=None:
        plt.savefig(save, dpi=120)
    else:
        plt.show()
    plt.close()


def show_batch_fakes(real_A, real_B, fake_A, fake_B, save_path):
    """
    批量显示并保存图像：每行 4 列 [real_A, fake_B, real_B, fake_A]
    输入 shape: (B, H, W, C)
    """
    B = real_A.shape[0]  # 批次大小
    cols = 4  # 固定 4 列
    rows = B  # 批次多少张，就多少行

    # 创建画布
    plt.figure(figsize=(cols * 4, rows * 4))  # 尺寸可按需要调整

    for idx in range(B):
        # 第 idx 行的 4 张图
        img_real_A = real_A[idx]
        img_fake_B = fake_B[idx]
        img_real_B = real_B[idx]
        img_fake_A = fake_A[idx]

        # 把数值限制在 [0,1]，防止显示异常
        img_real_A = np.clip(img_real_A, 0, 1)
        img_fake_B = np.clip(img_fake_B, 0, 1)
        img_real_B = np.clip(img_real_B, 0, 1)
        img_fake_A = np.clip(img_fake_A, 0, 1)

        # --------------------- 第 idx 行：4 列 ---------------------
        # 列 1：real_A
        plt.subplot(rows, cols, idx * cols + 1)
        plt.imshow(img_real_A)
        plt.axis("off")
        plt.title("Real A", fontsize=12)

        # 列 2：fake_B (A→B)
        plt.subplot(rows, cols, idx * cols + 2)
        plt.imshow(img_fake_B)
        plt.axis("off")
        plt.title("Fake B (A→B)", fontsize=12)

        # 列 3：real_B
        plt.subplot(rows, cols, idx * cols + 3)
        plt.imshow(img_real_B)
        plt.axis("off")
        plt.title("Real B", fontsize=12)

        # 列 4：fake_A (B→A)
        plt.subplot(rows, cols, idx * cols + 4)
        plt.imshow(img_fake_A)
        plt.axis("off")
        plt.title("Fake A (B→A)", fontsize=12)

    # 紧凑布局 + 保存
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()  # 释放内存



def append_to_txt(text: str, file_path: str = "log.txt"):
    """
    将字符串追加写入到txt文件，自动换行，不会覆盖原有内容

    Args:
        text (str): 要写入的字符串
        file_path (str): 保存的txt文件路径，默认 log.txt
    """
    try:
        # a = 追加模式；encoding=utf-8 保证中文不乱码
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")  # 自动换行
    except Exception as e:
        print(f"写入文件失败：{e}")


'''
历史图像缓冲区
不只用最新生成的图训练判别器
而是存最近 50 张生成图，随机抽图给判别器学习
好处：防止判别器过拟合、防止训练震荡
'''
class ImageBuffer:
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        self.buffer = []

    def query(self, image):
        if self.buffer_size == 0:
            return image

        result = []
        for img in image:
            img = img.unsqueeze(0)
            # 缓冲区未满 → 直接存
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(img)
                result.append(img)
            # 缓冲区满 → 50% 概率用历史图，50% 用新图
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, self.buffer_size, (1,)).item()
                    result.append(self.buffer[idx])
                    self.buffer[idx] = img
                else:
                    result.append(img)
        return torch.cat(result, dim=0)