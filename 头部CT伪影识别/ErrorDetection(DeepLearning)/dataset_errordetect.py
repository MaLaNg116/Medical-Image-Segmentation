import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import sklearn.utils
from collections import Counter
import pydicom
import cv2
import os

random_seed = 234
# -------------------------------#
#          图像预处理方法
# -------------------------------#
# 可选：图像阈值范围缩减
def windows_pro(img, min_bound=0, max_bound=85):
    img[img > max_bound] = max_bound
    img[img < min_bound] = min_bound
    img = img - min_bound
    img = normalize(img)
    return img


# 可选：直方图均衡(增加对比度)
def equalize_hist(img):
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    return img


# 必选：缩放尺寸，默认缩放为224
def img_resize(img, size=224):
    img = cv2.resize(img, (size, size))
    return img


# 必选：归一化
def normalize(img):
    img = img.astype(np.float32)
    np.seterr(divide='ignore', invalid='ignore')
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255
    img = img.astype(np.uint8)
    return img


# 必选：扩展为3通道
def extend_channels(img):
    img_channels = np.zeros([img.shape[0], img.shape[1], 3])
    img_channels[:, :, 0] = img
    img_channels[:, :, 1] = img
    img_channels[:, :, 2] = img
    return img_channels


# 图像预处理（伪影增强）
def data_preprocess_enhanced(img, size):
    img = equalize_hist(img)
    img = img_resize(img, size)
    img = normalize(img)
    img = extend_channels(img)
    img = img.astype(np.uint8)
    return img


# -------------------------------#
#       读取数据并划分数据集
# -------------------------------#
def data_load(path, size):
    dicomlist = []
    labels = []
    images = []


    with open(path, "r") as f:
        for line in f.readlines():
            img_path = line.strip().split(',')[0]
            dicomlist.append(img_path)
            label = line.strip().split(',')[1]
            label = '0' if label == 'good' else '1'
            labels.append(label)


    labels = np.array(labels)
    images = np.array([data_preprocess_enhanced(pydicom.read_file(dcm).pixel_array, size) for dcm in dicomlist])

    class_0_images = images[labels == '0']
    class_0_labels = labels[labels == '0']
    class_1_images = images[labels == '1']
    class_1_labels = labels[labels == '1']



    class_0_train_img, class_0_val_img, class_0_train_label, class_0_val_label = train_test_split(
        class_0_images, class_0_labels, test_size=0.2, random_state=random_seed)

    # 直接将类别1数据用于测试集
    test_img = images
    test_label = labels
    train_img = class_0_train_img
    train_label = class_0_train_label
    val_img = class_0_val_img
    val_label = class_0_val_label

    train_img, train_label = sklearn.utils.shuffle(train_img, train_label, random_state=random_seed)
    val_img, val_label = sklearn.utils.shuffle(val_img, val_label, random_state=random_seed)
    test_img, test_label = sklearn.utils.shuffle(test_img, test_label, random_state=random_seed)

    return train_img, train_label, val_img, val_label, test_img, test_label,class_0_images,class_0_labels,\
           class_1_images,class_1_labels


def mkdir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


class Dataset(data.Dataset):
    def __init__(self, img, label, transform=None):
        self.img = img
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img = self.img[index]
        target = int(self.label[index])
        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.img)


def get_anomaly_dataset(path, size, batch_size):
    train_img, train_label, val_img, val_label, test_img, test_label,class_0_images,class_0_labels,\
           class_1_images,class_1_labels = data_load(path, size)


    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((0, 180), expand=False),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    train_set = Dataset(train_img, train_label,train_data_transform)

    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    val_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    val_set = Dataset(val_img, val_label, val_data_transform)
    val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    test_data_transform = transforms.Compose([
        transforms.ToTensor()])
    test_set = Dataset(test_img, test_label, test_data_transform)
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    class1_set = Dataset(class_1_images,class_1_labels,test_data_transform)
    class1_loader = data.DataLoader(dataset=class1_set, batch_size=batch_size, shuffle=False)

    class0_set = Dataset(class_0_images,class_0_labels,test_data_transform)
    class0_loader = data.DataLoader(dataset=class0_set, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader,class0_loader,class1_loader




