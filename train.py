#!/usr/bin/python3

import argparse
import itertools
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from models import Generator,Discriminator
from utils import ReplayBuffer,LambdaLR,append_to_txt,batch_tensor_to_image,\
    show_batch_fakes,ImageBuffer,tensor_to_image
from datasets import ImageDataset



# ====================== 1. 超参数（严格对应论文）======================
img_size = 256      # 输入图像分辨率 256x256
img_channels = 3    # 3通道RGB
batch_size = 4      # 论文 batch size = 1
lr = 0.0002         # 论文学习率
epoch_num = 200     # 论文总轮次 200
lambda_cyc = 10     # 循环损失权重 λ=10（论文3.3节）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decay_epoch=100     # 多少轮次后学习率线性降低
offset=0            # 用于中断后再次加载模型训练，如果训练到100epoch后中断，这里就得填写100
lambda_idt = 0.5  # 身份损失权重，论文中取0.5（仅当训练非对称任务时启用）
dataroot=r"H:\datasets\Horse2Zebra\horse2zebra" # 文件路径，文件结构如下所示
# horse2zebra/
#   ├─trainA/
#   ├     xxx.jpg
#   ├─testB/
#   ├─trainB/
#   └─testA/



# ====================== 4. 损失函数 (论文 3 公式化描述) ======================
# GAN 损失（论文使用最小二乘损失 LSGAN） GAN 损失（对抗损失）：让生成图像看起来真实
gan_loss = nn.MSELoss().to(device)
# 循环一致性损失（论文 L1 损失） 循环一致性损失（Cycle Loss）：保证图像翻译后不丢失信息
cycle_loss = nn.L1Loss().to(device)

# ====================== 5. 初始化模型 ======================
G_A2B = Generator(img_channels, img_channels).to(device)  # A→B
G_B2A = Generator(img_channels, img_channels).to(device)  # B→A
D_A = Discriminator(img_channels).to(device)             # 判断A真假
D_B = Discriminator(img_channels).to(device)             # 判断B真假

# 历史图像缓冲区（论文：size=50）
buffer_A = ImageBuffer(buffer_size=50)
buffer_B = ImageBuffer(buffer_size=50)

# 优化器（论文 Adam 优化器）
optimizer_G = optim.Adam(
    itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
    lr=lr, betas=(0.5, 0.999)
)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(epoch_num, offset, decay_epoch).step)
lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(epoch_num, offset, decay_epoch).step)
lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(epoch_num, offset, decay_epoch).step)


# ====================== 7. 训练循环 ======================
def train_epoch(dataloader,epoch,save_dir,save_pth):
    '''

    :param dataloader:数据加载器
    :param epoch: 轮次数  int
    :param save_dir: 保存的路径 string
    :param save_pth: 是否保存 bool
    :return:
    '''
    G_A2B.train()
    G_B2A.train()
    D_A.train()
    D_B.train()

    lst_loss_G_A2B=[]
    lst_loss_G_B2A=[]
    lst_loss_cycle_A=[]
    lst_loss_cycle_B=[]
    lst_loss_idt_A = []
    lst_loss_idt_B = []
    lst_loss_real_A=[]
    lst_loss_fake_A=[]
    lst_loss_real_B=[]
    lst_loss_fake_B=[]


    for (real_A, real_B) in tqdm(dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # ---------------------- 训练生成器 G_A2B, G_B2A ----------------------
        optimizer_G.zero_grad()

        # 0. 身份映射损失 (论文 3.2 节 Identity Loss)
        # G_A2B 是 A→B（斑马→马），那你把真实的马（real_B）喂给 G_A2B，它应该 “几乎不动”，直接输出马本身。 这就是身份映射损失的意义！
        idt_B = G_A2B(real_B)  # 输入B到G_A2B，应输出接近B的图像
        idt_A = G_B2A(real_A)  # 输入A到G_B2A，应输出接近A的图像
        loss_idt_B = cycle_loss(idt_B, real_B)  # 复用L1损失（和循环损失一致）
        loss_idt_A = cycle_loss(idt_A, real_A)
        total_idt_loss = loss_idt_A + loss_idt_B

        # 1. 对抗损失 (论文 3.1 节) 让生成器生成的假图像骗过判别器。
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        loss_G_A2B = gan_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)).to(device))
        loss_G_B2A = gan_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)).to(device))

        # 2. 循环一致性损失 (论文 3.2 节) 保证 x -> y -> x' ≈ x
        rec_A = G_B2A(fake_B)
        rec_B = G_A2B(fake_A)

        loss_cycle_A = cycle_loss(rec_A, real_A)
        loss_cycle_B = cycle_loss(rec_B, real_B)
        total_cycle_loss = loss_cycle_A + loss_cycle_B

        # 总损失 (论文 3.3 节)
        loss_G = loss_G_A2B + loss_G_B2A + lambda_cyc * total_cycle_loss + lambda_idt * total_idt_loss
        loss_G.backward()
        optimizer_G.step()

        # ---------------------- 训练判别器 D_A ----------------------
        optimizer_D_A.zero_grad()
        loss_real_A = gan_loss(D_A(real_A), torch.ones_like(D_A(real_A)).to(device))
        fake_A_buffer = buffer_A.query(fake_A.detach())
        loss_fake_A = gan_loss(D_A(fake_A_buffer), torch.zeros_like(D_A(fake_A_buffer)).to(device))
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # ---------------------- 训练判别器 D_B ----------------------
        optimizer_D_B.zero_grad()
        loss_real_B = gan_loss(D_B(real_B), torch.ones_like(D_B(real_B)).to(device))
        fake_B_buffer = buffer_B.query(fake_B.detach())
        loss_fake_B = gan_loss(D_B(fake_B_buffer), torch.zeros_like(D_B(fake_B_buffer)).to(device))
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        # 将损失存储到列表中
        lst_loss_G_A2B.append(loss_G_A2B.item())
        lst_loss_G_B2A.append(loss_G_B2A.item())
        lst_loss_cycle_A.append(loss_cycle_A.item())
        lst_loss_cycle_B.append(loss_cycle_B.item())
        lst_loss_idt_A.append(loss_idt_A.item())
        lst_loss_idt_B.append(loss_idt_B.item())
        lst_loss_real_A.append(loss_real_A.item())
        lst_loss_fake_A.append(loss_fake_A.item())
        lst_loss_real_B.append(loss_real_B.item())
        lst_loss_fake_B.append(loss_fake_B.item())

    lr_G = optimizer_G.param_groups[0]['lr']
    lr_D_A = optimizer_D_A.param_groups[0]['lr']
    lr_D_B = optimizer_D_B.param_groups[0]['lr']

    # 更新学习率
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if save_pth:

        os.makedirs(save_dir, exist_ok=True)
        # 保存 4 个模型 + 2 个优化器（方便断点续训）
        torch.save(G_A2B.state_dict(), os.path.join(save_dir, f"G_A2B_epoch_{epoch}.pth"))
        torch.save(G_B2A.state_dict(), os.path.join(save_dir, f"G_B2A_epoch_{epoch}.pth"))
        torch.save(D_A.state_dict(), os.path.join(save_dir, f"D_A_epoch_{epoch}.pth"))
        torch.save(D_B.state_dict(), os.path.join(save_dir, f"D_B_epoch_{epoch}.pth"))

        # 可选：保存优化器（推荐，方便继续训练）
        torch.save(optimizer_G.state_dict(), os.path.join(save_dir, f"optimizer_G_epoch_{epoch}.pth"))
        torch.save(optimizer_D_A.state_dict(), os.path.join(save_dir, f"optimizer_D_A_epoch_{epoch}.pth"))
        torch.save(optimizer_D_B.state_dict(), os.path.join(save_dir, f"optimizer_D_B_epoch_{epoch}.pth"))

    return [
        lr_G,lr_D_A,lr_D_B,
        np.mean(lst_loss_G_A2B),
        np.mean(lst_loss_G_B2A),
        np.mean(lst_loss_cycle_A),
        np.mean(lst_loss_cycle_B),
        np.mean(lst_loss_idt_A),
        np.mean(lst_loss_idt_B),
        np.mean(lst_loss_real_A),
        np.mean(lst_loss_fake_A),
        np.mean(lst_loss_real_B),
        np.mean(lst_loss_fake_B),
    ]


def valid_epoch(dataloader, epoch, save_dir="./valid_results"):
    """
    验证函数：处理测试集第一个batch，执行A↔B风格迁移并保存结果

    Args:
        dataloader: 测试集dataloader
        epoch: 当前训练轮次（用于文件名）
        save_dir: 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 设置生成器为评估模式
    G_A2B.eval()
    G_B2A.eval()

    with torch.no_grad():  # 禁用梯度计算，节省内存
        # 获取测试集第一个batch
        for batch_idx, (real_A, real_B) in enumerate(dataloader):
            # 将数据移到指定设备
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # 执行风格迁移：A→B 和 B→A
            fake_B = G_A2B(real_A)  # A域（马）→ B域（斑马）
            fake_A = G_B2A(real_B)  # B域（斑马）→ A域（马）

            # 将张量转换为可显示的图像格式
            real_A_imgs = batch_tensor_to_image(real_A)
            real_B_imgs = batch_tensor_to_image(real_B)
            fake_B_imgs = batch_tensor_to_image(fake_B)
            fake_A_imgs = batch_tensor_to_image(fake_A)

            # 保存可视化结果
            save_path = os.path.join(save_dir, f"epoch_{epoch}_valid_batch_{batch_idx}.png")
            show_batch_fakes(
                real_A=real_A_imgs,
                real_B=real_B_imgs,
                fake_A=fake_A_imgs,
                fake_B=fake_B_imgs,
                save_path=save_path
            )
            print(f"验证结果已保存至: {save_path}")



def load_weight(load_epoch=100):
    '''
    :param load_epoch: 加载的pth轮次数，会读取./train_pth/{load_epoch}下的所有pth文件并加载
    :return:
    '''
    # ====================== 加载已保存的权重（直接插入使用）======================
    # 加载路径（改成你实际的 epoch 数字）
    load_dir = f"./train_pth/{load_epoch}"
    # 加载生成器 + 判别器权重
    G_A2B.load_state_dict(torch.load(os.path.join(load_dir, f"G_A2B_epoch_{load_epoch}.pth"), map_location=device))
    G_B2A.load_state_dict(torch.load(os.path.join(load_dir, f"G_B2A_epoch_{load_epoch}.pth"), map_location=device))
    D_A.load_state_dict(torch.load(os.path.join(load_dir, f"D_A_epoch_{load_epoch}.pth"), map_location=device))
    D_B.load_state_dict(torch.load(os.path.join(load_dir, f"D_B_epoch_{load_epoch}.pth"), map_location=device))
    # 【可选】加载优化器权重（断点续训必须加）
    optimizer_G.load_state_dict(torch.load(os.path.join(load_dir, f"optimizer_G_epoch_{load_epoch}.pth"), map_location=device))
    optimizer_D_A.load_state_dict(torch.load(os.path.join(load_dir, f"optimizer_D_A_epoch_{load_epoch}.pth"), map_location=device))
    optimizer_D_B.load_state_dict(torch.load(os.path.join(load_dir, f"optimizer_D_B_epoch_{load_epoch}.pth"), map_location=device))
    # 加载后必须设为训练模式
    G_A2B.train()
    G_B2A.train()
    D_A.train()
    D_B.train()
    print('权重已加载',load_dir)


def test_image(model, img_path):
    '''
    :param model:推理使用的模型，可以是G_A2B也可以是G_B2A
    :param img_path: 输入的图像路径，例如 ./img1.jpg
    :return: 模型的输出张量，shape=[1,3,256,256]
    '''
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        fake = model(img)
    return fake


def train():
    '''
    主训练函数
    :return:
    '''

    transforms_ = [transforms.Resize(img_size),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader_train = DataLoader(
        ImageDataset(root=dataroot, transforms_=transforms_, unaligned=True, mode="train"),  # 读取数据集 trainA和trainB下的所有图像
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    dataloader_test = DataLoader(
        ImageDataset(root=dataroot, transforms_=transforms_, unaligned=True, mode="test"),  # 读取数据集 testA和testB下的所有图像
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    print('数据已加载')

    print('训练开始')

    '''
    offset就是轮次偏移量，初始训练就是0，中断训练填写最新保存的epoch数值
    '''
    for epoch in range(offset, epoch_num):
        #                                                                                               这里要注意，0 5 10 15 20 25 ... 会保存权重
        lst_info = train_epoch(dataloader=dataloader_train, epoch=epoch, save_dir=f"./train_pth/{epoch}",save_pth=(epoch % 5 == 0))
        append_to_txt(f"epoch={epoch},{str(lst_info)}", "Train_Info.txt")
        valid_epoch(dataloader=dataloader_test, epoch=epoch, save_dir=f"./valid_results/{epoch}")



'''
推理测试集并保存
'''
def valid():
    transforms_ = [transforms.Resize(img_size),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader_test = DataLoader(
        ImageDataset(root=dataroot, transforms_=transforms_, unaligned=True, mode="test"),  # 读取数据集 testA和testB下的所有图像
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    EPO=230
    load_weight(load_epoch=EPO)
    valid_epoch(dataloader_test,0,f"./infer_results/{EPO}")



'''
推理函数
'''
def inference():

    load_weight(load_epoch=230)
    #  img_path  填写你想要推理的图像，G_A2B是把马变成斑马，所以img_path应该是一张256x256分辨率3通道的马图像
    fake_b = test_image(model=G_A2B,img_path=r"E:\桌面\马.jpg")
    np_fakeb = tensor_to_image(fake_b[0])
    plt.imshow(np_fakeb)
    plt.show()

if __name__ == '__main__':


    '''
    写给读者：
    
    我提供了一份预训练权重，使用3090显卡训练的，训练了250轮次，每轮次大约费时5分钟。
    最后马转为斑马效果还凑合，斑马转为马效果不咋地，可能是训练损失权重需要调整，论文中该数据集图像示例我训练的模型大多都能达到和论文类似效果
    可能作者也是挑一些好的展示到论文中，由于跑模型太费钱，我也一穷二白，项目就到这里了，权重您要用得上就用吧，
    这个项目我就是复现试验一下。
    如果有实在不会的地方可以提交issue或加我QQ：2648239704，当然我可能比较忙，望理解。
    
    祝您实验好运！
    '''

    # train()
    # valid()
    inference()




