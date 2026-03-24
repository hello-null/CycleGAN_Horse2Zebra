# CycleGAN_Horse2Zebra
基于CycleGAN的马与斑马无配对图像风格迁移模型 | 含完整训练代码 + 预训练权重

## 项目简介
本项目实现经典无监督图像风格迁移任务：马 ↔ 斑马互相转换。
提供完整训练、推理、可视化代码，以及训练好的模型权重（.pth），可直接运行使用。

## 项目结构，代码里有注释，仔细察看
infer_results：保存的测试集推理结果

train_pth：保存的模型权重

cyclegan_train_visualization.png：本人实验的训练日志可视化

datasets.py：数据集加载器

models.py：模型代码

train.py：主训练代码，也包含验证和推理

TrainInfo.txt：我的训练日志

utils.py：一些公共函数库

visual.py：将TrainInfo.txt可视化，保存为cyclegan_train_visualization.png

## 效果展示
### 训练损失可视化
![CycleGAN训练损失曲线](https://github.com/hello-null/CycleGAN_Horse2Zebra/blob/main/cyclegan_train_visualization.png)

### 马 ↔ 斑马 风格迁移结果 [马->斑马效果好些，斑马->马效果差]
![马 ↔ 斑马](https://github.com/hello-null/CycleGAN_Horse2Zebra/blob/main//infer_results/230/epoch_0_valid_batch_0.png)

![马 ↔ 斑马](https://github.com/hello-null/CycleGAN_Horse2Zebra/blob/main//infer_results/230/epoch_0_valid_batch_1.png)

![马 ↔ 斑马](https://github.com/hello-null/CycleGAN_Horse2Zebra/blob/main//infer_results/230/epoch_0_valid_batch_2.png)

![马 ↔ 斑马](https://github.com/hello-null/CycleGAN_Horse2Zebra/blob/main//infer_results/230/epoch_0_valid_batch_3.png)

![马 ↔ 斑马](https://github.com/hello-null/CycleGAN_Horse2Zebra/blob/main//infer_results/230/epoch_0_valid_batch_4.png)
