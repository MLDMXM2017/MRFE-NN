import torch
import torchvision
from torchsummary import summary
#import torch_densenet_cbam
import mynet_twoway
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import random
from sklearn.manifold import TSNE


# 为使训练结果可复现 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)


def train_test_split_two_way(data_1,data_2, label, rate):
    index = int(len(data_1)*rate)
    train_data_1 = data_1[0:index]
    train_data_2 = data_2[0:index]
    train_label = label[0:index]
    test_data_1 = data_1[index:]
    test_data_2 = data_2[index:]
    test_label = label[index:]
    return train_data_1, train_data_2, train_label, test_data_1, test_data_2, test_label
# 定义绘制混淆矩阵的函数
"""
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.GnBu):
    # 定义类别标签
    labels = ['coarse_towel', 'crocodile_pattern', 'relief_cloth', 'sponge', 'horizontal_fabric','linen', 'diamond_pattern']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
"""
# tick_marks = np.array(range(len(labels))) + 0.5
def cm_plot(y_true,y_pred,save_name):
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    #plt.figure(figsize=(12, 8), dpi=120)

    #ind_array = np.arange(len(labels))
    #x, y = np.meshgrid(ind_array, ind_array)

    #for x_val, y_val in zip(x.flatten(), y.flatten()):
    #    c = cm_normalized[y_val][x_val] # 报错提示index out of size 7
    #    if c > 0.01:
    #        plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=10, va='center', ha='center')
    # offset the tick
    #plt.gca().set_xticks(tick_marks, minor=True)
    #plt.gca().set_yticks(tick_marks, minor=True)
    #plt.gca().xaxis.set_ticks_position('none')
    #plt.gca().yaxis.set_ticks_position('none')
    #plt.grid(True, which='minor', linestyle='-')
    #plt.gcf().subplots_adjust(bottom=0.15)
"""
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig(model_path+save_name+'.png', format='png')
    plt.show()
"""    
#　定义模型存储路径
#model_path = './cross_data_eaxm/batch2_train/'
    
#model = cbam_densenet.DenseNet(input_shape=(44,44,1), nb_classes=7, depth=60, growth_rate=12,
#                          dropout_rate=0.1, bottleneck=False, compression=0.5).build_model()
#model.summary()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = mynet_twoway.myNet(channels_in=1, compression=1, growth_rate=1, num_classes=7,num_bottleneck=[2, 2, 2],
                        num_channels_before_dense=24,
                        num_dense_block=3).to(device)
#model = torch_densenet_cbam.DenseNet(channels_in=1, compression=0.5, growth_rate=12, num_classes=8,num_bottleneck=[18, 18, 18],
#                        num_channels_before_dense=24,
#                        num_dense_block=3).to(device)
#summary(model, [(1, 44, 44),(1, 44, 44)])
summary(model, (1, 44, 44))


#data_transform = transforms.Compose([
#    transforms.Resize((44,44)),
#    transforms.ToTensor(),
#])
# root='/home/yqh/workplace/DeepDG/datautil/sense_data/sense_data/'
# domain_name = 'batch1'
# data_dir = root + domain_name

# 这个函数加载出来的图像矩阵都是三通道的（3,44,44），并且没有什么参数调用可以让其变为单通道
# train_dataset = torchvision.datasets.ImageFolder(data_dir,data_transform)
# 因此，选择写numpy数组再转为pytorch的Tensor()张量
# print(train_dataset.classes)
# print(train_dataset.class_to_idx)
# print(train_dataset.imgs)
# print(train_dataset[0]) # 表示取第一个训练样本，即(path， class_index)。
# print(train_dataset[0][0]) # 返回的数据是PIL Image对象

# 读取数据
data_X_2_5_7_10_batch2 = np.load('./cross_data/data_X_2_5_7_10_batch1_7class.npy')
data_y_2_5_7_10_batch2 = np.load('./cross_data/data_y_2_5_7_10_batch1_7class.npy')

data_X_2_5_7_10_batch3 = np.load('./cross_data/data_X_2_5_7_10_batch2_7class.npy')
data_y_2_5_7_10_batch3 = np.load('./cross_data/data_y_2_5_7_10_batch2_7class.npy')

data_X_2_5_7_10_batch1 = np.load('./cross_data/data_X_series_2_4_6_8_7class.npy')
data_y_2_5_7_10_batch1 = np.load('./cross_data/data_y_series_2_4_6_8_7class.npy')

data_X_2_5_7_10_batch1 = data_X_2_5_7_10_batch1[:, np.newaxis, :, :] # np.newaxis 增加一维
data_X_2_5_7_10_batch2 = data_X_2_5_7_10_batch2.squeeze()[ :, np.newaxis, :, :] # np.newaxis 增加一维
data_X_2_5_7_10_batch3 = data_X_2_5_7_10_batch3.squeeze()[:, np.newaxis, :, :] # np.newaxis 增加一维

print(data_X_2_5_7_10_batch1.shape)
print(data_y_2_5_7_10_batch1.shape)
print(data_y_2_5_7_10_batch1[0:100])

print(data_X_2_5_7_10_batch2.shape)
print(data_y_2_5_7_10_batch2.shape)
print(data_y_2_5_7_10_batch2[0:100])

print(data_X_2_5_7_10_batch3.shape)
print(data_y_2_5_7_10_batch3.shape)
print(data_y_2_5_7_10_batch3[:100])

# 制作dataset
data_x = torch.from_numpy(data_X_2_5_7_10_batch1)
data_y = torch.from_numpy(data_y_2_5_7_10_batch1)

data_x = data_x.type(torch.FloatTensor)
data_y = torch.LongTensor(data_y.numpy())

dataset = torch.utils.data.TensorDataset(data_x,data_y)
# 制作测试集

test_x_1 = torch.from_numpy(data_X_2_5_7_10_batch1)
test_y_1 = torch.from_numpy(data_y_2_5_7_10_batch1)

test_x_1 = test_x_1.type(torch.FloatTensor)
test_y_1 = torch.LongTensor(test_y_1.numpy())

test_dataset_1 = torch.utils.data.TensorDataset(test_x_1,test_y_1)

test_x_2 = torch.from_numpy(data_X_2_5_7_10_batch2)
test_y_2 = torch.from_numpy(data_y_2_5_7_10_batch2)

test_x_2 = test_x_2.type(torch.FloatTensor)
test_y_2 = torch.LongTensor(test_y_2.numpy())

test_dataset_2 = torch.utils.data.TensorDataset(test_x_2,test_y_2)

test_x_3 = torch.from_numpy(data_X_2_5_7_10_batch3)
test_y_3 = torch.from_numpy(data_y_2_5_7_10_batch3)

test_x_3 = test_x_3.type(torch.FloatTensor)
test_y_3 = torch.LongTensor(test_y_3.numpy())

test_dataset_3 = torch.utils.data.TensorDataset(test_x_3,test_y_3)

# 划分训练集为train + val dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 装载数据
train_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4) 
val_dataloders = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True,num_workers=4) 


test_dataloders_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=64, shuffle=True,num_workers=4) 
test_dataloders_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=64, shuffle=True,num_workers=4) 
test_dataloders_3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=64, shuffle=True,num_workers=4) 

# 神经网络结构
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)   # 学习率为0.001, L2 penalty weight_decay 原1e-4
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15,gamma=0.1)
criterion = nn.CrossEntropyLoss()   # 损失函数也可以自己定义，我们这里用的交叉熵损失函数

# enumerate是python的内置函数，包括索引+数据
#train, train_lbp = list(enumerate(train_dataloders)), list(enumerate(train_lbp_dataloders))
#train_steps = min(len(train), len(train_lbp))


print("Start Training")
# 训练部分
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# 设定一个max_acc，验证过程中比较，保存使val_acc最大的模型
max_acc = 0.001
for epoch in range(30):    # 训练的数据量为5个epoch，每个epoch为一个循环
    print('epoch {}'.format(epoch + 1))
    train_loss = 0
    train_acc = 0
    for i, data in enumerate(train_dataloders, 0):  
    #for batch_idx in range(train_steps):  
        # enumerate是python的内置函数，既获得索引也获得数据
        # get the inputs
        inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
        inputs, labels = inputs.to(device), labels.to(device)
        # wrap them in Variable
        #inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable
        
        # forward + backward + optimize
        #outputs = model(inputs)        # 把数据输进网络
        # 多特征输入，LBP作为辅助特征
        outputs = model(inputs)        
        loss = criterion(outputs, labels)  # 计算损失值
        
        pred = torch.max(outputs, 1)[1]
        
        train_correct = (pred == labels).sum()
        train_acc += train_correct.item()
        
        optimizer.zero_grad()        # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
        loss.backward()                    # loss反向传播
        scheduler.step(loss) #监督loss mode为min，当loss停止下降时，更改lr动态衰减
        optimizer.step()                   # 反向传播后参数更新 
        
        train_loss += loss.item()       # loss累加
    # scheduler.step(train_loss / (len(train_dataset))) #监督loss mode为min，当loss停止下降时，更改lr动态衰减    
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_dataset)), train_acc / (len(train_dataset))))
    train_loss_list.append(train_loss / (len(train_dataset)))
    train_acc_list.append(train_acc / (len(train_dataset)))
    # evaluation--------------------------------
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for i, data in enumerate(val_dataloders, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        eval_loss += loss.item()
        pred = torch.max(outputs, 1)[1]
        
        num_correct = (pred == labels).sum()
        eval_acc += num_correct.item()
    temp_acc = eval_acc / (len(val_dataset))
    #print(temp_acc,max_acc)
    if temp_acc > max_acc:
        max_acc = temp_acc
        print("save beset model,max_acc=",max_acc)
        torch.save(model.state_dict(),'mynet_batch1_MRF_spa_coord_twoway.pth')
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_dataset)), eval_acc / (len(val_dataset))))
    val_loss_list.append(eval_loss / (len(val_dataset)))
    val_acc_list.append(eval_acc / (len(val_dataset)))
    
    #print(train_acc_list,val_acc_list,train_loss_list,val_loss_list)
print('Finished Training')

# 保存神经网络
# torch.save(model, 'model.pkl')                      # 保存整个神经网络的结构和模型参数
# torch.save(model.state_dict(), 'model_params.pkl')  # 只保存神经网络的模型参数
'''
# 绘制训练 & 验证的准确率值
plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
'''
# 加载保存的最佳模型
model.load_state_dict(torch.load("mynet_batch1_MRF_spa_coord_twoway.pth"))
'''
# 模型预测（在全新的跨批次数据集上）
model.eval()
test_acc_1 = 0
pred_list_1 = []
labels_list_1 = []
output_np_1 = np.empty(shape=[0, 7])
for i, data in enumerate(test_dataloders_1, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    labels_list = labels.cpu().numpy().tolist()
    labels_list_1.extend(labels_list)        
    with torch.no_grad():  
        outputs = model(inputs)
        output_np = outputs.cpu().numpy()
        output_np_1 = np.append(output_np_1, output_np,axis=0)
    pred = torch.max(outputs, 1)[1]
    pred_list = pred.cpu().numpy().tolist()
    pred_list_1.extend(pred_list)
    
    num_correct = (pred == labels).sum()
    test_acc_1 += num_correct.item()
test_acc_1 = test_acc_1 / (len(test_dataset_1))
print("batch1_acc:",test_acc_1)
# 绘制预测结果混淆矩阵
cm_plot(labels_list_1, pred_list_1, "mynet_batch2_pred_batch_1")


'''
model.eval()
test_acc_2 = 0
pred_list_2 = []
labels_list_2 = []
output_np_2 = np.empty(shape=[0, 7])
for i, data in enumerate(test_dataloders_2, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    labels_list = labels.cpu().numpy().tolist()
    labels_list_2.extend(labels_list)        
    with torch.no_grad(): 
        outputs = model(inputs)
        output_np = outputs.cpu().numpy()
        output_np_2 = np.append(output_np_2, output_np,axis=0)
    pred = torch.max(outputs, 1)[1]
    pred_list = pred.cpu().numpy().tolist()
    pred_list_2.extend(pred_list)
    
    num_correct = (pred == labels).sum()
    test_acc_2 += num_correct.item()
test_acc_2 = test_acc_2 / (len(test_dataset_2))
print("batch2_acc:",test_acc_2)
# 绘制预测结果混淆矩阵
cm_plot(labels_list_2, pred_list_2, "mynet_batch1_pred_batch_2")

model.eval()
test_acc_3 = 0
pred_list_3 = []
labels_list_3 = []
output_np_3 = np.empty(shape=[0, 7])
for i, data in enumerate(test_dataloders_3, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    labels_list = labels.cpu().numpy().tolist()
    labels_list_3.extend(labels_list)    
    with torch.no_grad(): 
        outputs = model(inputs)
        output_np = outputs.cpu().numpy()
        output_np_3= np.append(output_np_3, output_np,axis=0)
    pred = torch.max(outputs, 1)[1]
    pred_list = pred.cpu().numpy().tolist()
    pred_list_3.extend(pred_list)
    
    num_correct = (pred == labels).sum()
    test_acc_3 += num_correct.item()
test_acc_3 = test_acc_3 / (len(test_dataset_3))
print("batch3_acc:",test_acc_3)
# 绘制预测结果混淆矩阵
cm_plot(labels_list_3, pred_list_3, "mynet_batch2_pred_batch_3")


'''
# 将得到的预测结果进行tsne降维，降至2维
tsne = TSNE(n_components=2,init = 'pca',random_state=0)
X_tsne = tsne.fit_transform(output_np_1)

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=labels_list_1,marker='.')
plt.colorbar()
plt.savefig('tsne_2_pred_1_source_only_dot_all'+'.jpg') 
plt.show()
'''

'''




# 将得到的预测结果进行tsne降维，降至2维
tsne = TSNE(n_components=2,init = 'pca',random_state=0)
X_tsne = tsne.fit_transform(output_np_2)

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=labels_list_2,marker='.')
plt.colorbar()
plt.show()

'''
'''
# 定义钩子
import cv2

# 类的作用
# 1.编写梯度获取hook
# 2.网络层上注册hook
# 3.运行网络forward backward
# 4.根据梯度和特征输出热力图
class ShowGradCam:
    def __init__(self,conv_layer):
        assert isinstance(conv_layer,torch.nn.Module), "input layer should be torch.nn.Module"
        self.conv_layer = conv_layer
        self.conv_layer.register_forward_hook(self.farward_hook)
        self.conv_layer.register_backward_hook(self.backward_hook)
        self.grad_res = []
        self.feature_res = []

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_res.append(grad_out[0].detach())

    def farward_hook(self,module, input, output):
        self.feature_res.append(output)

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (44, 44))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def show_on_img(self,input_img):
        #write heatmap on target img
        #:param input_img: cv2:ndarray/img_pth
        #:return: save jpg
        if isinstance(input_img,str):
            input_img = cv2.imread(input_img)
        input_img = input_img.squeeze()
        print('input_img:',input_img.shape)
        img_size = (input_img.shape[1],input_img.shape[0])
        #img_size = (input_img.shape[2],input_img.shape[1])
        fmap = self.feature_res[0].cpu().data.numpy().squeeze()
        grads_val = self.grad_res[0].cpu().data.numpy().squeeze()
        cam = self.gen_cam(fmap, grads_val)
        print('cam:',cam.shape)
        cam = cv2.resize(cam, img_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)/255.
        print('heatmap:',heatmap.shape)
        input_img = cv2.cvtColor(input_img.numpy(), cv2.COLOR_GRAY2BGR)
        cam = heatmap + np.float32(input_img/255.)
        cam = cam / np.max(cam)*255
        cv2.imwrite('tsne_1_pred_2/so_horizontal/oringin_horizontal_ours_4017.jpg',input_img)
        cv2.imwrite('tsne_1_pred_2/so_horizontal/grad_feature_horizontal_so_4017.jpg',cam)
        print('save gradcam result in grad_feature.jpg')
        
def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,(44, 44))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input

def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec
print(data_y_2_5_7_10_batch2[4015:4025])# index0 3 7 9 13 17 23 38 241 252 257 425 550 1249 2394 2397 2508 2510 2511 2519 3611 3629 3935是纱布类别6 -> 4
#(1,44,44)->(1,1,44,44)    
test_img = data_X_2_5_7_10_batch2[4017][np.newaxis,:, :, :] # np.newaxis 增加一维.unsqueeze(0)
test_img = torch.from_numpy(test_img)
test_y = torch.from_numpy(data_y_2_5_7_10_batch2[4017][np.newaxis])
print(test_img.shape,test_y)
test_img = test_img.type(torch.FloatTensor)
test_y = torch.LongTensor(test_y.numpy())
test_img, test_y = test_img.to(device), test_y.to(device)

gradCam = ShowGradCam(model.dense_3)

output = model(test_img)
pred = torch.max(output, 1)[1]
print('labels:',test_y)
print('pred:',pred)
# backward
model.zero_grad()
class_loss = criterion(output, test_y)
class_loss.backward()
print(test_img.shape)
# save result
gradCam.show_on_img(test_img.cpu())
'''
'''
# b1训练预测1_3 源域b1，目标域b3 
model.eval()
test_acc_val = 0
pred_list_val = []
labels_list_val = []
output_np_val = np.empty(shape=[0, 8])
for i, data in enumerate(test_dataloders_1_3, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    labels_list = labels.cpu().numpy().tolist()
    labels_list_val.extend(labels_list)    
    with torch.no_grad(): 
        outputs = model(inputs)
        output_np = outputs.cpu().numpy()
        output_np_val= np.append(output_np_val, output_np,axis=0)
    pred = torch.max(outputs, 1)[1]
    pred_list = pred.cpu().numpy().tolist()
    pred_list_val.extend(pred_list)
    
    num_correct = (pred == labels).sum()
    test_acc_val += num_correct.item()
test_acc_val = test_acc_val / (len(test_dataset_1_3))
print("batch1_val_acc:",test_acc_val)
# 绘制预测结果混淆矩阵
cm_plot(labels_list_val, pred_list_val, "torch_cbam_densenet_batch_2_pred_batch_3")
# 将得到的预测结果进行tsne降维，降至2维
tsne = TSNE(n_components=2,init = 'pca',random_state=0)
X_tsne = tsne.fit_transform(output_np_val)

# 设置colorbar的颜色
cm1 = plt.cm.get_cmap('viridis') 
for j in range(output_np_val.shape[0]):  
    if j < 1920:
        if labels_list_val[j] ==0 :
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c='#355d8a',marker='+')
        elif labels_list_val[j] ==1:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='x')
        elif labels_list_val[j] ==2:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='*')
        elif labels_list_val[j] ==3:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='<')
        elif labels_list_val[j] ==4:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='p')
        elif labels_list_val[j] ==5:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='d')
        elif labels_list_val[j] ==6:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='s')
        elif labels_list_val[j] ==7:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='h')
    else:
        #plt.scatter(X_tsne[j,0],X_tsne[j,1],c=labels_list_val[j],marker='x') 
        if labels_list_val[j] ==0 :
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c='#4cc070',marker='+')
        elif labels_list_val[j] ==1:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='x')
        elif labels_list_val[j] ==2:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='*')
        elif labels_list_val[j] ==3:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='<')
        elif labels_list_val[j] ==4:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='p')
        elif labels_list_val[j] ==5:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='d')
        elif labels_list_val[j] ==6:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='s')
        elif labels_list_val[j] ==7:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='h')  
#plt.scatter(X_tsne[:,0],X_tsne[:,1],c=labels_list_val,marker='.')

#plt.legend((s1,s2),('source domain:Batch 1','target domain:Batch 3') ,loc = 'best')
plt.colorbar()
plt.savefig('tsne_1_pred_1_3_source_only_dot_all_diff_marker'+'.jpg') 
plt.show()
'''
'''
model.eval()
test_acc_3 = 0
pred_list_3 = []
labels_list_3 = []
output_np_3 = np.empty(shape=[0, 8])
for i, data in enumerate(test_dataloders_3, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    labels_list = labels.cpu().numpy().tolist()
    labels_list_3.extend(labels_list)    
    with torch.no_grad(): 
        outputs = model(inputs)
        output_np = outputs.cpu().numpy()
        output_np_3= np.append(output_np_3, output_np,axis=0)
    pred = torch.max(outputs, 1)[1]
    pred_list = pred.cpu().numpy().tolist()
    pred_list_3.extend(pred_list)
    
    num_correct = (pred == labels).sum()
    test_acc_3 += num_correct.item()
test_acc_3 = test_acc_3 / (len(test_dataset_3))
print("batch3_acc:",test_acc_3)
# 绘制预测结果混淆矩阵
cm_plot(labels_list_3, pred_list_3, "torch_cbam_densenet_batch_2_pred_batch_3")
# 将得到的预测结果进行tsne降维，降至2维
tsne = TSNE(n_components=2,init = 'pca',random_state=0)
X_tsne = tsne.fit_transform(output_np_3)

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=labels_list_3,marker='.')
plt.colorbar()
plt.savefig('tsne_1_2_pred_3_source_only_dot_all'+'.jpg') 
plt.show()

'''
'''
if labels_list_val[j] ==0 :
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c='#470151',marker='x')
        elif labels_list_val[j] ==1:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#47307f",marker='x')
        elif labels_list_val[j] ==2:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#355d8a",marker='x')
        elif labels_list_val[j] ==3:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#287e8e",marker='x')
        elif labels_list_val[j] ==4:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#20a387",marker='x')
        elif labels_list_val[j] ==5:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#4cc070",marker='x')
        elif labels_list_val[j] ==6:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#9fdb41",marker='x')
        elif labels_list_val[j] ==7:
            plt.scatter(X_tsne[j,0],X_tsne[j,1],c="#ffe327",marker='x') 
'''