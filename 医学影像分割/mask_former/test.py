import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from model import VGG_CrossAttentionDecoder  # Assuming this is the defined model
from dataset import ImageDataset  # Assuming this is the dataset class
import argparse

# 分类函数
def category(img):
    if img[0] >= 0.5:
        return 1
    elif img[1] >= 0.5:
        return 2
    elif img[2] >= 0.5:
        return 3
    else:
        return 0


# 计算像素精度PA
def pixel_accuracy(label, predict):
    start_time = time.time()
    true_pixel = (label == predict).sum().item()
    all_pixels = label.numel()
    accuracy = true_pixel / all_pixels
    end_time = time.time()
    print(f"the pixel_accuracy is: {accuracy}")
    print(f"compute pixel_accuracy use time: {end_time - start_time}")
    return accuracy


# 计算均相似精度MPA
def mean_pixel_accuracy(label, predict, class_num=4):
    start_time = time.time()
    class_list = np.zeros(class_num)
    intersection_list = np.zeros(class_num)

    for n in range(class_num):
        class_list[n] = (label == n).sum().item()
        intersection_list[n] = ((label == n) & (predict == n)).sum().item()

    mpa = np.mean(intersection_list / (class_list + 1e-6))  # Avoid divide by zero
    end_time = time.time()
    print(f"the mean pixel accuracy is: {mpa}")
    print(f"compute mean pixel accuracy use time: {end_time - start_time}")
    return mpa


# 计算均交并比mIoU
def compute_mIoU(label, predict, class_num=4):
    start_time = time.time()
    class_list = np.zeros(class_num)
    intersection_list = np.zeros(class_num)

    for n in range(class_num):
        class_list[n] = ((label == n) | (predict == n)).sum().item()
        intersection_list[n] = ((label == n) & (predict == n)).sum().item()

    mIoU = np.mean(intersection_list / (class_list + 1e-6))  # Avoid divide by zero
    end_time = time.time()
    print(f"the mIoU is: {mIoU}")
    print(f"compute mIoU use time: {end_time - start_time}")
    return mIoU


# 将输出的张量转为图像
def tensorToimg(img):
    img = img.argmax(dim=0).cpu().numpy()  # 将one-hot输出转为类别
    return img

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="mix_data/", required=False, help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0001')
    parser.add_argument('--model-save', default='./models/best_model.pth', help='folder to output model checkpoints')
    parser.add_argument('--model-path', default='./models/best_model.pth',
                        help='folder of model checkpoints to predict')
    parser.add_argument('--outf', default="./test/test-mix", required=False, help='path of predict output')
    args = parser.parse_args(args=[])
    return args


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


class DiceLoss(nn.Module):
    def forward(self, y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 模型推理
import numpy as np

def predict_level3(test_dir=""):
    args = get_parser()  # 获取参数
    test_dataset = ImageDataset(args.data_root, "test/" + test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = VGG_CrossAttentionDecoder()
    model.load_state_dict(torch.load('models/nf_mix_model.pth'))
    model = model.to(device)
    model.eval()

    dice_scores, pa_scores, mpa_scores, mIoU_scores = [], [], [], []

    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(img)

        # 计算Dice系数
        dice = dice_coefficient(label, output)
        print(f"the dice coefficient is: {dice.item()}")
        dice_scores.append(dice.item())

        # 计算像素精度、MPA和mIoU
        pred_img = (output > 0.5).float().cpu()  # 二值化预测图像
        label_img = label.cpu()

        pa = pixel_accuracy(label_img, pred_img)
        mpa = mean_pixel_accuracy(label_img, pred_img)
        mIoU = compute_mIoU(label_img, pred_img)

        pa_scores.append(pa)
        mpa_scores.append(mpa)
        mIoU_scores.append(mIoU)

        # 显示与保存结果
        ori_img = img.squeeze().cpu().numpy()  # 原始图像
        ori_gt = label.squeeze(0).cpu().numpy()  # 真实标签
        final_img = pred_img.squeeze().cpu().numpy()  # 预测掩码

        # 将图像转换为适合 imshow 的格式
        if ori_img.ndim == 3:  # 如果原始图像是 RGB 格式 (C, H, W)
            ori_img = np.transpose(ori_img, (1, 2, 0))  # 转换为 (H, W, C)
        if ori_gt.ndim == 3:  # 如果真实标签是 (C, H, W)
            ori_gt = np.transpose(ori_gt, (1, 2, 0))  # 转换为 (H, W, C)
        if final_img.ndim == 3:  # 如果预测图像是 (C, H, W)
            final_img = np.transpose(final_img, (1, 2, 0))  # 转换为 (H, W, C)

        plt.figure(figsize=(12, 4))  # 调整图像大小
        plt.subplot(1, 3, 1)
        plt.imshow(ori_img, cmap='gray')
        plt.axis('off')
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(ori_gt, cmap='gray')
        plt.axis('off')
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')
        plt.title(f"Predicted\nDice: {dice:.4f}")

        plt.savefig(f"{args.outf}/{i}.png")
        plt.close()

    print(f"Average Dice: {np.mean(dice_scores):.4f}")
    print(f"Average Pixel Accuracy: {np.mean(pa_scores):.4f}")
    print(f"Average MPA: {np.mean(mpa_scores):.4f}")
    print(f"Average mIoU: {np.mean(mIoU_scores):.4f}")


# 主函数
if __name__ == "__main__":
    s_t = time.time()
    predict_level3(test_dir="")
    # print("================CVC-300====================")
    # predict_level3(test_dir="CVC-300")
    # print("===========================================")

    # print("=============CVC-ClinicDB==================")
    # predict_level3(test_dir="CVC-ClinicDB")
    # print("===========================================")

    # print("==============CVC-ColonDB==================")
    # predict_level3(test_dir="CVC-ColonDB")
    # print("===========================================")
    #
    # print("===========ETIS-LaribPolypDB===============")
    # predict_level3(test_dir="ETIS-LaribPolypDB")
    # print("===========================================")
    #
    # print("================Kvasir=====================")
    # predict_level3(test_dir="Kvasir")
    # print("===========================================")

    print("Total time:", time.time() - s_t)
