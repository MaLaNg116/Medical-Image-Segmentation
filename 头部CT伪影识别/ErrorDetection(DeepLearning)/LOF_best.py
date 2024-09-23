import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, accuracy_score, confusion_matrix,
                             precision_recall_curve, roc_curve, auc, f1_score)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns  # 用于绘制热图
import torch
import dataset_errordetect as dataset
import VGG_KCN
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# 参数设置
is_train = False  # True-训练模型  False-测试模型
is_pretrained = True  # 是否加载预训练权重
backbone = 'vgg16_no_freeze'  # 骨干网络：alexnet resnet18 vgg16 densenet inception
model_path = 'model/errordetect_KCN/' + backbone  # 模型存储路径

# 训练参数设置
SIZE = 299 if backbone == 'inception' else 224  # 图像进入网络的大小
BATCH_SIZE = 16  # batch_size数
NUM_CLASS = 1  # 分类数
EPOCHS = 30  # 迭代次数

# 进入工程路径并新建文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 进入工程路径
dataset.mkdir('model')  # 新建文件夹
best_model_name = os.path.join(model_path, 'L3_%s_best_model.pth' % backbone)
PATH = 'data/labels.csv'
TEST_PATH = 'data/labels_orginal.csv'
model = VGG_KCN.VGGKAN().to(device)
if not is_train:
    best_model = torch.load(best_model_name)
    model.load_state_dict(best_model, False)
    print(f"Model path: {best_model_name}")

# 加载数据
dataset.mkdir(model_path)
train_loader, val_loader, all_train_loader, _, _ = dataset.get_anomaly_dataset(PATH, SIZE, BATCH_SIZE)
_, _, test_loader, class0_loader, class1_loader = dataset.get_anomaly_dataset(TEST_PATH, SIZE, BATCH_SIZE)

# 准备存储测试数据和标签
y_test = []
y_pred = []

# 切换模型到评估模式
model.eval()

# 从 DataLoader 中提取测试数据
with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)

        # 模型预测
        outputs = model(data).cpu().numpy()
        predictions = (outputs > 0.2).astype(int)  # 阈值设定为0.5，1表示异常，0表示正常

        y_pred.extend(predictions)  # 模型预测标签
        y_test.extend(labels.cpu().numpy())  # 真实标签

y_test = np.array(y_test)
y_pred = np.array(y_pred)

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
X_test_pca_2d = pca.fit_transform(np.vstack([data.cpu().view(data.size(0), -1).numpy() for data, _ in test_loader]))

# 创建子图
fig = plt.figure(figsize=(15, 6))

# 二维可视化
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X_test_pca_2d[y_test == 0, 0], X_test_pca_2d[y_test == 0, 1], label='Normal', alpha=0.5, color='blue')
ax1.scatter(X_test_pca_2d[y_test == 1, 0], X_test_pca_2d[y_test == 1, 1], label='True Anomaly', alpha=0.5, color='red')
# Modify the scatter call to correctly index the 2D array based on predictions
anomaly_indices = np.where(np.array(y_pred) == 1)[0]  # Get the indices where prediction is 1 (anomaly)
ax1.scatter(X_test_pca_2d[anomaly_indices, 0], X_test_pca_2d[anomaly_indices, 1], label='Predicted Anomaly', alpha=0.5, s=100, marker='x', color='black')

ax1.set_title('PCA of Test Data (2D)')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.legend()
ax1.grid()

# PCA 降维到3维
pca = PCA(n_components=3)
X_test_pca_3d = pca.fit_transform(np.vstack([data.cpu().view(data.size(0), -1).numpy() for data, _ in test_loader]))

# 三维可视化
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter3D(X_test_pca_3d[y_test == 0, 0], X_test_pca_3d[y_test == 0, 1], X_test_pca_3d[y_test == 0, 2],
              label='Normal', alpha=0.5, color='blue')
ax2.scatter3D(X_test_pca_3d[y_test == 1, 0], X_test_pca_3d[y_test == 1, 1], X_test_pca_3d[y_test == 1, 2],
              label='True Anomaly', alpha=0.5, color='red')
# Modify the 3D scatter call to correctly index the array
anomaly_indices = np.where(np.array(y_pred) == 1)[0]  # Get the indices where prediction is 1 (anomaly)
ax2.scatter3D(X_test_pca_3d[anomaly_indices, 0], X_test_pca_3d[anomaly_indices, 1], X_test_pca_3d[anomaly_indices, 2],
              label='Predicted Anomaly', alpha=0.5, s=100, marker='x', color='black')


ax2.set_title('PCA of Test Data (3D)')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax2.set_zlabel('Principal Component 3')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

# 绘制 Precision-Recall 曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()

# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_pred)
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
