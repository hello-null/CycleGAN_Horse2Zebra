import glob
import random
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

"""
CycleGAN 专用无配对/配对图像数据集加载类（内存预加载版）
用于加载两个域（A域 和 B域）的图像数据，支持配对与非配对训练模式。
【优化】初始化时一次性将所有图像加载到内存，训练时直接从内存读取
"""
class ImageDataset(Dataset):

    '''
    初始化数据集路径、预处理方法、加载模式
    【关键修改】初始化时预加载所有图像到内存

    Args:
        root (str): 数据集根目录路径
        transforms_ (list): 图像预处理变换列表（会被 compose 封装）
        unaligned (bool): 是否使用非配对数据加载方式，True=随机取B，False=按索引对应取B
        mode (str): 数据集模式，可选 'train' / 'test'，对应加载 trainA/trainB 或 testA/testB文件夹中的图像
    '''
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        path_A = os.path.join(root, mode+'A')
        path_B = os.path.join(root, mode+'B')

        # 检查路径是否存在
        if not os.path.exists(path_A):
            raise ValueError(f"Directory not found: {path_A}")
        if not os.path.exists(path_B):
            raise ValueError(f"Directory not found: {path_B}")

        # 获取文件列表
        self.files_A = sorted(glob.glob(os.path.join(path_A, '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(path_B, '*.*')))

        # 检查是否找到文件
        if len(self.files_A) == 0:
            raise ValueError(f"No files found in {path_A}. Please check the dataset path.")
        if len(self.files_B) == 0:
            raise ValueError(f"No files found in {path_B}. Please check the dataset path.")

        # ===================== 核心修改 =====================
        # 初始化时一次性将所有图像加载到内存，只执行一次
        print(f"Loading {mode} dataset into memory...")
        self.images_A = self._load_images_to_memory(self.files_A)
        self.images_B = self._load_images_to_memory(self.files_B)
        print(f"Loaded {len(self.images_A)} A images, {len(self.images_B)} B images into memory.")
        # ====================================================

    def _load_images_to_memory(self, file_list):
        """
        工具函数：将文件列表中的所有图像一次性读取到内存列表中
        """
        images = []
        for file_path in file_list:
            # 读取图像并转为RGB，存入列表
            img = Image.open(file_path).convert('RGB')
            images.append(img)
        return images

    """
    根据索引获取一组配对（A,B）图像数据
    【修改】直接从内存列表取图像，不再读取磁盘
    item_A是马   item_B是斑马
    """
    def __getitem__(self, index):
        # 直接从内存取A域图像
        item_A = self.transform(self.images_A[index % len(self.images_A)])

        if self.unaligned:
            # 非配对：随机从内存取B
            item_B = self.transform(self.images_B[random.randint(0, len(self.images_B) - 1)])
        else:
            # 配对：按索引从内存取B
            item_B = self.transform(self.images_B[index % len(self.images_B)])

        return (item_A, item_B)

    """
    返回数据集长度，取A、B域中较大的长度
    """
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



if __name__ == '__main__':

    # 用于测试
    transforms_ = [transforms.Resize(int(256 * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(256),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = DataLoader(
        ImageDataset(r"H:\datasets\Horse2Zebra\horse2zebra", transforms_=transforms_, unaligned=True),
        batch_size=16,
        shuffle=True,
        num_workers=1
    )

    for i, (real_A,real_B) in enumerate(dataloader):
        print(i,real_A.shape,real_B.shape)
        exit()