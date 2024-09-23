import torch.optim
from pylab import *
import os
import dataset_errordetect as dataset
# import KCN
import VGG_KCN
import time
from sklearn import metrics
start111 = time.time()


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start = datetime.datetime.now()
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


# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=6e-3)  # 更新所有层权重
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-1)
criterion = torch.nn.MSELoss()


if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

train_batch = 40


# train_resnet
def train_resnet(model):
    history_train = []
    history_valid = []
    best_valid_loss = float('inf')  # Initialize the best valid loss

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
            if batch_index < train_batch:
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                model.train()
                optimizer.zero_grad()
                output = model(batch_x).squeeze()  # Ensure output is (batch_size,)
                loss = criterion(output, batch_y.float())  # Convert batch_y to float
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_train_loss = total_loss / (batch_index + 1)
        print('[Epoch=%d/%d]Train set: Avg_loss=%.4f' % (epoch + 1, EPOCHS, avg_train_loss))
        history_train.append(avg_train_loss)

        scheduler.step()

        avg_valid_loss, valid_rmse = valid_resnet(model)

        print('[Epoch=%d/%d]Validation set: Avg_loss=%.4f, RMSE=%.4f' % (
            epoch + 1, EPOCHS, avg_valid_loss, valid_rmse))
        history_valid.append(avg_valid_loss)

        # 保存最优模型（基于验证集损失）
        if avg_valid_loss < best_valid_loss and epoch >= 2:
            print('>>>>>>>>>>>>>>Best model is %s' % (str(epoch + 1) + '.pkl'))
            torch.save(model.state_dict(), best_model_name)
            best_valid_loss = avg_valid_loss

    print("Train finished!")
    print('Train running time = %s' % str(datetime.datetime.now() - start))
    print('Saving last model...')
    last_model_name = os.path.join(model_path, 'L3_%s_last_model.pth' % backbone)
    torch.save(model.state_dict(), last_model_name)

    return best_model_name, history_train, history_valid


def valid_resnet(model):
    with torch.no_grad():
        model.eval()
        val_loss_list = []
        for batch_index, (batch_valid_x, batch_valid_y) in enumerate(val_loader, 0):
            if torch.cuda.is_available():
                batch_valid_x, batch_valid_y = batch_valid_x.cuda(), batch_valid_y.cuda()
            output = model(batch_valid_x).squeeze()
            loss = criterion(output, batch_valid_y.float())  # Convert batch_valid_y to float
            val_loss_list.append(loss.item())

        avg_valid_loss = np.mean(val_loss_list)
        valid_rmse = np.sqrt(avg_valid_loss)
        return avg_valid_loss, valid_rmse

if is_train:
    # 训练集
    best_model_name, history_train, history_valid = train_resnet(model)


def for_test_resnet():
    print('------ Data Evaluation Start ------')
    val_loss_list = []
    predictions = []
    true_values = []

    with torch.no_grad():
        model.eval()
        for val_x, val_y in test_loader:
            if torch.cuda.is_available():
                images, labels = val_x.cuda(), val_y.cuda()
            else:
                images, labels = val_x, val_y

            output = model(images)

            loss = criterion(output.squeeze(), labels)

            val_loss_list.append(loss.item())
            predictions.extend(output.squeeze().cpu().numpy())
            true_values.extend(labels.cpu().numpy())


    # Convert predictions and true values to numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)


    return true_values, predictions


# Function to compute metrics and print in a specific format
def compute_metrics(true_values, predictions, threshold=0.2):
    # Convert predictions to binary labels based on the threshold
    binary_predictions = (np.abs(predictions-true_values) > threshold).astype(int)  # 0: normal, 1: anomaly
    binary_true_values = true_values.astype(int)  # Based on true class (0/1)

    # Calculate accuracy
    acc = 100 * metrics.accuracy_score(binary_true_values, binary_predictions)

    # Calculate AUC using predictions (directly from regression output)
    abs_diff = np.abs(predictions - true_values)
    scores = np.exp(abs_diff)
    max_score = np.max(scores)
    min_score = np.min(scores)
    scores_normalized = (scores - min_score) / (max_score - min_score)
    scores_normalized = np.clip(scores_normalized, 0, 1)

    auc = metrics.roc_auc_score(y_true=binary_true_values, y_score=scores_normalized )

    # Classification report
    class_report = metrics.classification_report(binary_true_values, binary_predictions, digits=4)

    # Confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(binary_true_values, binary_predictions).ravel()

    # Print metrics in the desired format
    print('Classification Report:\n', class_report)
    print('Accuracy of the network is: %.4f %%' % acc)
    print('AUC: %.4f' % auc)
    print('TN=%d, FP=%d, FN=%d, TP=%d' % (tn, fp, fn, tp))

    return acc, auc, tn, fp, fn, tp


# Run the test and print the results for the combined dataset
true_values, predictions = for_test_resnet()

print("----- Combined Data Results -----")
accuracy, auc, tn, fp, fn, tp = compute_metrics(true_values, predictions)






