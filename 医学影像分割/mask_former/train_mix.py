import torch
import torch.nn as nn
import torch.optim as optim
from model import VGG_CrossAttentionDecoder
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import time
import matplotlib.pyplot as plt
from dataset import ImageDataset


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


class DiceLoss(nn.Module):
    def forward(self, y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)


# 验证模型
def validate(model, val_loader, criterion, device):
    model.eval()
    total_dice = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for val_data, val_labels in val_loader:
            val_data = val_data.to(device)
            val_labels = val_labels.to(device)

            mask_pred = model(val_data)

            loss = criterion(val_labels, mask_pred)
            dice = dice_coefficient(val_labels, mask_pred)

            total_loss += loss.item()
            total_dice += dice.item()

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    return avg_loss, avg_dice


# 模型训练
def train_level3(freeze):
    args = get_parser()  # 获取参数

    # 通过 DataLoader 加载数据
    train_dataset = ImageDataset(args.data_root, "train")
    val_dataset = ImageDataset(args.data_root, "val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = VGG_CrossAttentionDecoder()
    model = model.to(device)

    # 如果需要冻结VGG模型的权重
    if freeze:
        print("Freezing VGG layers.")
        for param in model.backbone1.parameters():
            param.requires_grad = False
        for param in model.backbone2.parameters():
            param.requires_grad = False
        for param in model.backbone3.parameters():
            param.requires_grad = False
    else:
        print("VGG layers are trainable.")

    if os.path.exists(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()

    best_val_dice = 0.0
    train_losses = []
    val_dices = []
    for epoch in range(args.niter):
        model.train()

        total_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass on train data
            mask_pred = model(batch_data)

            # 计算损失
            loss = criterion(batch_labels, mask_pred)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation after each epoch
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        train_losses.append(avg_train_loss)
        val_dices.append(val_dice)

        print(
            f'Epoch [{epoch + 1}/{args.niter}], Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # Save best model based on validation dice coefficient
        if val_dice > best_val_dice:
            torch.save(model.state_dict(), args.model_path)
            best_val_dice = val_dice

    plot_history(train_losses, val_dices, args.outf)


# 绘制训练曲线
def plot_history(train_losses, val_dices, result_dir):
    plt.plot(train_losses, marker='.', color='r')
    plt.title('Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(os.path.join(result_dir, 'train_loss.png'))
    plt.show()

    plt.plot(val_dices, marker='.', color='b')
    plt.title('Validation Dice Coefficient')
    plt.xlabel('epoch')
    plt.ylabel('dice coefficient')
    plt.grid()
    plt.savefig(os.path.join(result_dir, 'val_dice.png'))
    plt.show()


# 获取命令行参数
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="mix_data", required=False, help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=16, help='input batch size')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0001')
    parser.add_argument('--model-save', default='./models/nf_mix_model.pth', help='folder to output model checkpoints')
    parser.add_argument('--model-path', default='./models/nf_mix_model.pth',
                        help='folder of model checkpoints to predict')
    parser.add_argument('--outf', default="./test/test-mix", required=False, help='path of predict output')
    args = parser.parse_args(args=[])
    return args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s_t = time.time()
train_level3(freeze=False)
print("Training time:", time.time() - s_t)
