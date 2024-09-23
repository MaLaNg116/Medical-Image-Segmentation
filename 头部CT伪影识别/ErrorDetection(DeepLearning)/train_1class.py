import torch.optim
from pylab import *
import os
import dataset_errordetect as dataset
import network_L3 as network
import time
start111 = time.time()



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start = datetime.datetime.now()
print('Device:', device)

# 参数设置
is_train = False  # True-训练模型  False-测试模型
is_pretrained = True  # 是否加载预训练权重
backbone = 'vgg16'  # 骨干网络：alexnet resnet18 vgg16 densenet inception
model_path = 'model/errordetect/' + backbone  # 模型存储路径

# 训练参数设置
SIZE = 299 if backbone == 'inception' else 224  # 图像进入网络的大小
BATCH_SIZE = 16  # batch_size数
NUM_CLASS = 1  # 分类数
EPOCHS = 15  # 迭代次数

# 进入工程路径并新建文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 进入工程路径
dataset.mkdir('model')  # 新建文件夹
best_model_name = os.path.join(model_path, 'L3_%s_best_model.pth' % backbone)
if is_train:  # 训练模式
    PATH = 'data/labels.csv'
    model = network.initialize_model(backbone, is_pretrained, NUM_CLASS)
else:  # 测试模型
    PATH = 'data/exam_labels.csv'
    # PATH = 'data/labels.csv'
    best_model = torch.load(best_model_name)
    model = network.initialize_model(backbone, False, NUM_CLASS)
    model.load_state_dict(best_model, False)
    print(f"Model path: {best_model_name}")


# 加载数据
dataset.mkdir(model_path)
train_loader, val_loader,test_loader = dataset.get_anomaly_dataset(PATH, SIZE, BATCH_SIZE)


# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=6e-3)  # 更新所有层权重
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,weight_decay=1e-1)
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


def for_test_resnet(positive, threshold=0.05):
    print('------ Validation Start ------')
    val_loss_list = []
    predictions = []
    true_values = []

    if not positive:
        loader = val_loader
    if positive:
        loader = test_loader
    with torch.no_grad():
        model.eval()
        for val_x, val_y in loader:
            if torch.cuda.is_available():
                images, labels = val_x.cuda(), val_y.cuda()
            else:
                images, labels = val_x, val_y
            output = model(images)
            loss = criterion(output.squeeze(), labels)
            val_loss_list.append(loss.item())
            predictions.extend(output.squeeze().cpu().numpy())
            true_values.extend(labels.cpu().numpy())
        # Calculate accuracy based on a threshold
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    accuracy = np.mean(np.abs(predictions - true_values) < threshold)  # Adjust accuracy calculation if needed
    if positive:
       accuracy = 1-accuracy

    return true_values, predictions, accuracy


# 在训练完成后调用测试函数
F_true, F_pred, F_accuracy = for_test_resnet(False)
T_true, T_pred, T_accuracy = for_test_resnet(True)
# # Save the entire validation results
# np.savez_compressed(os.path.join(model_path, 'validation_results.npz'),
#                     true_values=F_true,
#                     predictions=F_pred,
#                     accuracy=F_accuracy)

print('头部ct运行时间level3alexnet：', time.time()-start111)
print('正常召回率: %.4f' % F_accuracy)
print('患病召回率: %.4f' % T_accuracy)
