import torch
from torch.utils.data import Dataset
import os
import cv2


# 增强预处理函数
def image_process_enhanced(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
    enhanced_gray_img = cv2.equalizeHist(gray_img)  # 对灰度图像进行直方图均衡

    # 复制灰度图像到三个通道，使其与原始图像具有相同的通道数
    enhanced_img = cv2.merge((enhanced_gray_img, enhanced_gray_img, enhanced_gray_img))

    return enhanced_img


# 自定义 Dataset 类
class ImageDataset(Dataset):
    def __init__(self, root, data_type):
        self.image_path = os.path.join(root, data_type, "images")
        self.label_path = os.path.join(root, data_type, "masks")

        self.image_list = os.listdir(self.image_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_path, self.image_list[idx])
        label_file = os.path.join(self.label_path, self.image_list[idx])

        img = cv2.imread(image_file)
        label = cv2.imread(label_file)

        img = cv2.resize(img, (224, 224))
        label = cv2.resize(label, (224, 224))

        # img = image_process_enhanced(img)  # 使用灰度增强

        img = img / 255.0
        label = label / 255.0

        img = img.transpose(2, 0, 1)  # 转为 [C, H, W]
        label = label.transpose(2, 0, 1)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
