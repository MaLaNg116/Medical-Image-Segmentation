# 云南大学 2021级人工智能 医疗实训项目-期末仓库

+ 小组成员
> 20211060245 陈俊宏
> 20211060143 李俊杰
> 20211060203 周楷翔
> 20211120259 朱　荣
> 20211060007 宋成宇

## 一、头部CT伪影识别

项目结构
```sh
./头部CT伪影识别
├───ErrorDetection(DeepLearning)
│       dataset_errordetect.py
│       dataset_imbalance.py
│       KCN.py
│       KNN.py
│       LOF_best.py
│       train_1class.py
│       train_1class_KAN.py
│       VGG_KCN.py
│
├───ErrorDetection(LOF)
│       1.png
│       2.png
│       dataset_L2.py
│       LOF_best.py
│
└───VGG+KAN二分类
        dataset_L2.py
        KCN.py
        train.py
```
如需运行代码，请阅读代码数据处理有关内容，并将数据集正确存放到对应位置。

### 1. Classification With VGG16+KAN
正文

### 2. Error Detection With LOF(Local Outlier Factor)
正文

<p align="center" style="font-weight:bold;">LOF最终分类效果</p>  

![分类结果](./头部CT伪影识别/ErrorDetection(LOF)/1.png)

<p align="center" style="font-weight:bold;">LOF PCA降维数据分布展示</p>  

![PCA数据展示](./头部CT伪影识别/ErrorDetection(LOF)/2.png)

### 3. Error Detection With VGG16+KAN
正文

## 医学影像分割

### 1. 使用自己搭建（或预训练）的主干模型 + U-Net进行分割

在 [./医学影像分割/Level3.ipynb](./医学影像分割/Level3.ipynb) 文件中，已经包含以下内容的所有代码与运行结果：

+ Segmentation With VGG16+U-Net(Single Modal)
+ Segmentation With VGG16+U-Net(Multi Modal)
+ Segmentation With ResNet50+U-Net(Multi Modal)

如需运行 jupyter notebook 中的相关代码，请根据代码中数据处理相关部分，将需要的文件夹、文件结构以及数据集正确创建和存放到到对应位置。

### 2. 使用自己搭建（或预训练）的主干模型 + SAM进行分割