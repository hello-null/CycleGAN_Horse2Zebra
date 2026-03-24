import matplotlib.pyplot as plt
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

'''
您需要修改这里的路径，确保它是train.py输出的训练日志txt文件，在这个项目中我已经提供了我训练的日志文件。
该代码运行后会保存到 ./cyclegan_train_visualization.png 一个可视化文件，用于查看实验信息
注意每个图有y轴刻度限制，可能您需要改动。例如：“ax2.set_ylim(ymax=0.65)”
'''
LOG_DIR=r'./TrainInfo.txt'




# English Font Settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('CycleGAN Training Loss & Learning Rate Curve', fontsize=20, fontweight='bold', y=0.98)

# Parse log data
epochs = []
lr_G, lr_D_A, lr_D_B = [], [], []
loss_G_A2B, loss_G_B2A = [], []
loss_cycle_A, loss_cycle_B = [], []
loss_idt_A, loss_idt_B = [], []
loss_DA_real, loss_DA_fake, loss_DB_real, loss_DB_fake = [], [], [], []

with open(LOG_DIR, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        epoch_match = re.search(r'epoch=(\d+)', line)
        num_match = re.search(r'\[(.*?)\]', line)
        if epoch_match and num_match:
            epoch = int(epoch_match.group(1))
            nums = list(map(float, num_match.group(1).split(',')))
            epochs.append(epoch)
            lr_G.append(nums[0])
            lr_D_A.append(nums[1])
            lr_D_B.append(nums[2])
            loss_G_A2B.append(nums[3])
            loss_G_B2A.append(nums[4])
            loss_cycle_A.append(nums[5])
            loss_cycle_B.append(nums[6])
            loss_idt_A.append(nums[7])
            loss_idt_B.append(nums[8])
            loss_DA_real.append(nums[9])
            loss_DA_fake.append(nums[10])
            loss_DB_real.append(nums[11])
            loss_DB_fake.append(nums[12])

epochs = np.array(epochs)
lr_G = np.array(lr_G)
lr_D_A = np.array(lr_D_A)
lr_D_B = np.array(lr_D_B)
loss_G_A2B = np.array(loss_G_A2B)
loss_G_B2A = np.array(loss_G_B2A)
loss_cycle_A = np.array(loss_cycle_A)
loss_cycle_B = np.array(loss_cycle_B)
loss_idt_A = np.array(loss_idt_A)
loss_idt_B = np.array(loss_idt_B)
loss_DA_real = np.array(loss_DA_real)
loss_DA_fake = np.array(loss_DA_fake)
loss_DB_real = np.array(loss_DB_real)
loss_DB_fake = np.array(loss_DB_fake)

loss_D_A = (loss_DA_real + loss_DA_fake) * 0.5
loss_D_B = (loss_DB_real + loss_DB_fake) * 0.5

# Plot 1: Learning Rate
ax1 = axes[0, 0]
ax1.plot(epochs, lr_G, label='Generator LR', color='#e74c3c', linewidth=2, marker='.', markersize=4)
ax1.plot(epochs, lr_D_A, label='Discriminator A LR', color='#3498db', linewidth=2, marker='.', markersize=4)
ax1.plot(epochs, lr_D_B, label='Discriminator B LR', color='#2ecc71', linewidth=2, marker='.', markersize=4)
ax1.set_title('Learning Rate', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Learning Rate', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.tick_params(axis='both', labelsize=10)

# Plot 2: Generator Adversarial Loss
ax2 = axes[0, 1]
ax2.plot(epochs, loss_G_A2B, label='G_A2B Loss', color='#9b59b6', linewidth=2, marker='.', markersize=4)
ax2.plot(epochs, loss_G_B2A, label='G_B2A Loss', color='#f39c12', linewidth=2, marker='.', markersize=4)
ax2.set_title('Generator Adversarial Loss', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('GAN Loss', fontsize=12)
ax2.set_ylim(ymax=0.65)
ax2.legend(loc='best', fontsize=10)
ax2.tick_params(axis='both', labelsize=10)

# Plot 3: Cycle Consistency Loss
ax3 = axes[1, 0]
ax3.plot(epochs, loss_cycle_A, label='Cycle A Loss', color='#1abc9c', linewidth=2, marker='.', markersize=4)
ax3.plot(epochs, loss_cycle_B, label='Cycle B Loss', color='#e67e22', linewidth=2, marker='.', markersize=4)
ax3.set_title('Cycle Consistency Loss', fontsize=14, fontweight='bold', pad=10)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Cycle Loss', fontsize=12)
ax3.set_ylim(ymax=0.15)
ax3.legend(loc='best', fontsize=10)
ax3.tick_params(axis='both', labelsize=10)

# Plot 4: Identity Loss
ax4 = axes[1, 1]
ax4.plot(epochs, loss_idt_A, label='Identity A Loss', color='#34495e', linewidth=2, marker='.', markersize=4)
ax4.plot(epochs, loss_idt_B, label='Identity B Loss', color='#95a5a6', linewidth=2, marker='.', markersize=4)
ax4.set_title('Identity Mapping Loss', fontsize=14, fontweight='bold', pad=10)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Identity Loss', fontsize=12)
ax4.set_ylim(ymax=0.2)
ax4.legend(loc='best', fontsize=10)
ax4.tick_params(axis='both', labelsize=10)

# Plot 5: Discriminator Total Loss
ax5 = axes[2, 0]
ax5.plot(epochs, loss_D_A, label='Discriminator A Loss', color='#c0392b', linewidth=2, marker='.', markersize=4)
ax5.plot(epochs, loss_D_B, label='Discriminator B Loss', color='#27ae60', linewidth=2, marker='.', markersize=4)
ax5.set_title('Discriminator Total Loss', fontsize=14, fontweight='bold', pad=10)
ax5.set_xlabel('Epoch', fontsize=12)
ax5.set_ylabel('Discriminator Loss', fontsize=12)
ax5.set_ylim(ymax=0.3)
ax5.legend(loc='best', fontsize=10)
ax5.tick_params(axis='both', labelsize=10)

# Plot 6: Discriminator Real/Fake Loss
ax6 = axes[2, 1]
ax6.plot(epochs, loss_DA_real, label='D_A Real', color='#e74c3c', linewidth=1.5, linestyle='-', marker='.', markersize=3)
ax6.plot(epochs, loss_DA_fake, label='D_A Fake', color='#e74c3c', linewidth=1.5, linestyle='--', marker='.', markersize=3)
ax6.plot(epochs, loss_DB_real, label='D_B Real', color='#3498db', linewidth=1.5, linestyle='-', marker='.', markersize=3)
ax6.plot(epochs, loss_DB_fake, label='D_B Fake', color='#3498db', linewidth=1.5, linestyle='--', marker='.', markersize=3)
ax6.set_title('Discriminator Real / Fake Loss', fontsize=14, fontweight='bold', pad=10)
ax6.set_xlabel('Epoch', fontsize=12)
ax6.set_ylabel('Loss Value', fontsize=12)
ax6.set_ylim(ymax=0.5)
ax6.legend(loc='best', fontsize=9)
ax6.tick_params(axis='both', labelsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('cyclegan_train_visualization.png', dpi=300, bbox_inches='tight')
# plt.show()