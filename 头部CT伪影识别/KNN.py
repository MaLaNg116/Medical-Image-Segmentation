import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import torch.nn as nn
import dataset_errordetect as dataset
import VGG_KCN
# import KCN
from sklearn import metrics
import torch.optim
import warnings

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

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
TEST_PATH = 'data/exam_labels.csv'
# model = KCN.ResNetKAN().cuda()
model = VGG_KCN.VGGKAN().cuda()
if not is_train:
    best_model = torch.load(best_model_name)
    model.load_state_dict(best_model, False)
    print(f"Model path: {best_model_name}")



# 加载数据
dataset.mkdir(model_path)
train_loader, val_loader, all_train_loader, _ , _ = dataset.get_anomaly_dataset(PATH, SIZE, BATCH_SIZE)
_, _,test_loader,class0_loader,class1_loader = dataset.get_anomaly_dataset(TEST_PATH, SIZE, BATCH_SIZE)

# 初始化模型
num_ftrs = model.num_features
# model.kan1 = nn.Identity()
# model.kan2 = nn.Identity()

# 定义KNN分类器
k = 5  # 选择K值
knn = KNeighborsClassifier(n_neighbors=k)

test_pred = []
test_true = []

with torch.no_grad():
    model.eval()
    for test_x, test_y in test_loader:
        if torch.cuda.is_available():
            images, labels = test_x.cuda(), test_y.cuda()
        else:
            images, labels = test_x, test_y
        features = model.vgg16(images)
        # features = model.resnet(images)
        # 训练KNN模型
        knn.fit(features.detach().cpu().numpy(), labels.detach().cpu().numpy())

        # 进行预测
        y_pred = knn.predict(features.detach().cpu().numpy())

        test_pred = np.hstack((test_pred, y_pred))
        test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

# 计算准确率
test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
print(f"KNN表征质量测试的准确率为: {test_acc:.2f}")
