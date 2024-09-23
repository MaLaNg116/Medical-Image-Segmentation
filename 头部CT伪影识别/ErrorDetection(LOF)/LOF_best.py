import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, accuracy_score, confusion_matrix,
                             precision_recall_curve, roc_curve, auc, f1_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns  # 用于绘制热图
import torch
import dataset_L2 as dataset

# 设置随机种子
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# 训练参数设置
PATH = 'data/labels_1.csv'
TEST_PATH = 'data/exam_labels.csv'  # 测试数据路径
SIZE = 224  # 图像进入网络的大小
BATCH_SIZE = 1  # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 50  # 迭代次数
is_train = True
is_sampling = 'no_sample'
is_windows_pro = False
is_equalize_hist = True

# 加载数据
train_loader, val_loader, _ = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train,
                                                    is_sampling=is_sampling, is_windows_pro=is_windows_pro,
                                                    is_equalize_hist=is_equalize_hist)
_, _, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=False,
                                         is_sampling=is_sampling, is_windows_pro=is_windows_pro,
                                         is_equalize_hist=is_equalize_hist)

# 从 DataLoader 中提取训练数据
X_train = []
y_train = []
X_test = []
y_test = []

for data, labels in train_loader:
    data_np = data.view(data.size(0), -1).numpy()  # 将四维张量展平为二维数组
    X_train.append(data_np)
    y_train.append(labels.numpy())

for data, labels in test_loader:
    data_np = data.view(data.size(0), -1).numpy()  # 将四维张量展平为二维数组
    X_test.append(data_np)
    y_test.append(labels.numpy())

X_train = np.concatenate(X_train)  # 合并训练数据
y_train = np.concatenate(y_train)  # 合并训练标签
X_test = np.concatenate(X_test)  # 合并测试数据
y_test = np.concatenate(y_test)  # 合并测试标签

# 训练 Local Outlier Factor 模型
n_neighbors = 3
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.3, novelty=True)
lof.fit(X_train)

# 进行预测，获取 LOF 分数
lof_scores = lof.decision_function(X_test)

# 将分数小于 0.01 的标记为异常点
y_pred = [1 if score < 0.01 else 0 for score in lof_scores]

# 计算 Precision、Recall 和 Accuracy
precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)
accuracy = accuracy_score(y_test, y_pred)
f1_0 = f1_score(y_test, y_pred, pos_label=0)
f1_1 = f1_score(y_test, y_pred, pos_label=1)

# 输出结果
print(f'Class 0 - Precision: {precision_0:.2f}, Recall: {recall_0:.2f}, F1 Score: {f1_0:.2f}')
print(f'Class 1 - Precision: {precision_1:.2f}, Recall: {recall_1:.2f}, F1 Score: {f1_1:.2f}')
print(f'Overall Accuracy: {accuracy:.2f}')

# PCA 降维并可视化
pca = PCA(n_components=2)
X_test_pca_2d = pca.fit_transform(X_test)

# 创建子图
fig = plt.figure(figsize=(15, 6))

# 二维可视化
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X_test_pca_2d[y_test == 0, 0], X_test_pca_2d[y_test == 0, 1], label='Normal', alpha=0.5, color='blue')
ax1.scatter(X_test_pca_2d[y_test == 1, 0], X_test_pca_2d[y_test == 1, 1], label='True Anomaly', alpha=0.5, color='red')
ax1.scatter(X_test_pca_2d[np.array(y_pred) == 1, 0], X_test_pca_2d[np.array(y_pred) == 1, 1], label='Predicted Anomaly', alpha=0.5, s=100, marker='x', color='black')

ax1.set_title('PCA of Test Data (2D)')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.legend()
ax1.grid()

# PCA 降维到3维
pca = PCA(n_components=3)
X_test_pca_3d = pca.fit_transform(X_test)

# 三维可视化
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter3D(X_test_pca_3d[y_test == 0, 0], X_test_pca_3d[y_test == 0, 1], X_test_pca_3d[y_test == 0, 2], label='Normal', alpha=0.5, color='blue')
ax2.scatter3D(X_test_pca_3d[y_test == 1, 0], X_test_pca_3d[y_test == 1, 1], X_test_pca_3d[y_test == 1, 2], label='True Anomaly', alpha=0.5, color='red')
ax2.scatter3D(X_test_pca_3d[np.array(y_pred) == 1, 0], X_test_pca_3d[np.array(y_pred) == 1, 1], X_test_pca_3d[np.array(y_pred) == 1, 2], label='Predicted Anomaly', alpha=0.5, s=100, marker='x', color='black')

ax2.set_title('PCA of Test Data (3D)')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax2.set_zlabel('Principal Component 3')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()
# 绘制 Precision-Recall 曲线
precision, recall, _ = precision_recall_curve(y_test, -lof_scores)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()

# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, -lof_scores)
roc_auc = auc(fpr, tpr)
plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.text(0.6, 0.2, f'AUC = {roc_auc:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

