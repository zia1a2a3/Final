import torch
from torch import nn
from torch.autograd import Variable

import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time

from tqdm import tqdm

#from models.apl import *
from models.apl_plus import *

#batch_size定义了每次训练的样本数,num_epochs定义了训练的轮数。
batch_size = 900
num_epochs = 250




# load data
# 首先用scipy.io的loadmat函数读取了train_data.mat文件,
# 从中获取train_data_amp, train_activity_label和train_location_label数据。
# 将activity和location的标签数据在第1维(列)上拼接起来得到完整的标签train_label。
#
# 然后将train_data和train_label转换为PyTorch的张量(Tensor)格式,
# 并创建了TensorDataset对象train_dataset。
# 最后用DataLoader将dataset封装起来,指定了batch大小和shuffle参数,便于后续按批次读取数据。

data_amp = sio.loadmat('mydata/train_data_all.mat')
train_data_amp = data_amp['train_data_amp']
train_data = train_data_amp

train_activity_label = data_amp['train_activity_label']
train_location_label = data_amp['train_location_label']
train_label = np.concatenate((train_activity_label, train_location_label), 1)

num_train_instances = len(train_data)

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)




# 测试数据test_data_amp, test_activity_label, test_location_label的读取和处理过程与训练数据类似
data_amp = sio.loadmat('mydata/test_data_all.mat')
test_data_amp = data_amp['test_data_amp']
test_data = test_data_amp

test_activity_label = data_amp['test_activity_label']
test_location_label = data_amp['test_location_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



#创建神经网络模型，使用了自定义的ResNet网络结构,
# 其中block参数指定了残差块的类型为BasicBlock,
# layers参数指定了每个残差块的层数为[1,1,1,1],
# inchannel参数指定了输入数据的通道数为57。
aplnet = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], inchannel=57)
 #aplnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], inchannel=57)
# aplnet = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], inchannel=52)

# 创建好的aplnet实例随后被移到GPU上。
aplnet = aplnet.cuda()


#定义损失函数和优化器
#损失函数使用了交叉熵函数CrossEntropyLoss,同时指定size_average参数为False。
# 优化器使用了Adam,学习率设置为0.005。
# 同时还定义了一个学习率调度器scheduler,它会在训练过程的第10、20、30...个epoch时将学习率乘以gamma参数0.5,即每次减半。
criterion = nn.CrossEntropyLoss(size_average=False).cuda()

optimizer = torch.optim.Adam(aplnet.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                             140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                 gamma=0.5)
train_loss_act = np.zeros([num_epochs, 1])
train_loss_loc = np.zeros([num_epochs, 1])
test_loss_act = np.zeros([num_epochs, 1])
test_loss_loc = np.zeros([num_epochs, 1])
train_acc_act = np.zeros([num_epochs, 1])
train_acc_loc = np.zeros([num_epochs, 1])
test_acc_act = np.zeros([num_epochs, 1])
test_acc_loc = np.zeros([num_epochs, 1])#这是干什么？？？？？？？



#训练过程,最外层是epoch的循环
#在每个epoch开始时,调用scheduler.step()调整学习率,并将模型设置为训练模式。
for epoch in range(num_epochs):
    print('Epoch:', epoch)
    aplnet.train()
    scheduler.step()
    # for i, (samples, labels) in enumerate(train_data_loader):
    loss_x = 0
    loss_y = 0


    # 内层是训练数据的循环,使用了tqdm模块来显示进度条:
    #对于每个batch的数据,首先将其移至GPU,并将activity标签和location标签分别提取出来。
    #这里使用了Variable封装了Tensor,但在PyTorch 0.4以后的版本就不再需要了。
    for (samples, labels) in tqdm(train_data_loader):
        samplesV = Variable(samples.cuda())
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act.cuda())
        labelsV_loc = Variable(labels_loc.cuda())

        # Forward + Backward + Optimize
        #首先用optimizer.zero_grad()将梯度清零,然后将一个batch的数据samplesV输入aplnet,得到预测的activity和location标签。
        #用交叉熵函数分别计算activity和location的loss,两个loss相加得到总的loss。
        #然后执行loss.backward()进行反向传播计算梯度,再用optimizer.step()执行一次梯度下降更新网络参数。
        optimizer.zero_grad()
        predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)

        loss_act = criterion(predict_label_act, labelsV_act)
        loss_loc = criterion(predict_label_loc, labelsV_loc)

        loss = loss_act + loss_loc
        loss.backward()
        optimizer.step()

        #每个batch的activity loss和location loss会分别累加到loss_x和loss_y中
        loss_x += loss_act.item()
        loss_y += loss_loc.item()

        # loss.backward()
        # optimizer.step()

    train_loss_act[epoch] = loss_x / num_train_instances
    train_loss_loc[epoch] = loss_y / num_train_instances

    #当一个完整的epoch训练完成后,会在训练集上评估模型效果
    #将模型设为评估模式aplnet.eval(),用with torch.no_grad()禁用求导。
    #然后同训练过程一样,将数据分batches输入网络
    aplnet.eval()
    # loss_x = 0
    correct_train_act = 0
    correct_train_loc = 0
    for i, (samples, labels) in enumerate(train_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()

            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act.cuda())
            labelsV_loc = Variable(labels_loc.cuda())

            #得到预测结果predict_label_act和predict_label_loc后,
            #分别与真实标签labelsV_act和labelsV_loc比较,计算预测正确的样本数,累加到correct_train_act和correct_train_loc中。
            predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)

            prediction = predict_label_loc.data.max(1)[1]
            correct_train_loc += prediction.eq(labelsV_loc.data.long()).sum()

            prediction = predict_label_act.data.max(1)[1]
            correct_train_act += prediction.eq(labelsV_act.data.long()).sum()

            loss_act = criterion(predict_label_act, labelsV_act)
            loss_loc = criterion(predict_label_loc, labelsV_loc)
            # loss_x += loss.item()

    #所有训练数据评估完后,可以计算当前模型在训练集上的准确率
    print("Activity Training accuracy:", (100 * float(correct_train_act) / num_train_instances))
    print("Location Training accuracy:", (100 * float(correct_train_loc) / num_train_instances))

    #训练准确率存储在train_acc_act和train_acc_loc数组中, 以便后续展示训练曲线。
    train_acc_act[epoch] = 100 * float(correct_train_act) / num_train_instances
    train_acc_loc[epoch] = 100 * float(correct_train_loc) / num_train_instances

    #测试过程与训练评估过程类似, 遍历所有测试数据, 输入模型计算准确率和loss, 存储在相应的数组中
    trainacc_act = str(100 * float(correct_train_act) / num_train_instances)[0:6]
    trainacc_loc = str(100 * float(correct_train_loc) / num_train_instances)[0:6]

    loss_x = 0
    loss_y = 0
    correct_test_act = 0
    correct_test_loc = 0
    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act.cuda())
            labelsV_loc = Variable(labels_loc.cuda())

        predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)
        prediction = predict_label_act.data.max(1)[1]
        correct_test_act += prediction.eq(labelsV_act.data.long()).sum()

        prediction = predict_label_loc.data.max(1)[1]
        correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()

        loss_act = criterion(predict_label_act, labelsV_act)
        loss_loc = criterion(predict_label_loc, labelsV_loc)
        loss_x += loss_act.item()
        loss_y += loss_loc.item()

    print("Activity Test accuracy:", (100 * float(correct_test_act) / num_test_instances))
    print("Location Test accuracy:", (100 * float(correct_test_loc) / num_test_instances))

    test_loss_act[epoch] = loss_x / num_test_instances
    test_acc_act[epoch] = 100 * float(correct_test_act) / num_test_instances

    test_loss_loc[epoch] = loss_y / num_test_instances
    test_acc_loc[epoch] = 100 * float(correct_test_loc) / num_test_instances

    testacc_act = str(100 * float(correct_test_act) / num_test_instances)[0:6]
    testacc_loc = str(100 * float(correct_test_loc) / num_test_instances)[0:6]

    #最后是保存模型和统计数据
    #在第1个epoch时,将temp_test和temp_train初始化为当前的测试和训练准确率。
    #此后如果当前epoch的测试准确率高于之前的最高值temp_test,就用torch.save()保存当前的模型参数
    # 命名规则包含了epoch数、训练和测试准确率等信息。同时更新temp_test和temp_train为当前值。
    if epoch == 0:
        temp_test = correct_test_act
        temp_train = correct_train_act
    elif correct_test_act > temp_test:
        torch.save(aplnet, 'myweights/net1111epoch' + str(
            epoch) + 'Train' + trainacc_act + 'Test' + testacc_act + 'Train' + trainacc_loc + 'Test' + testacc_loc + '.pkl')

        temp_test = correct_test_act
        temp_train = correct_train_act


# for learning curves
#训练完所有epoch后,将训练和测试的loss、accuracy数据保存到.mat文件中
sio.savemat(
    'myresult/net1111TrainLossAct_Train' + str(100 * float(temp_train) / num_train_instances)[
                                                                 0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'train_loss': train_loss_act})
sio.savemat(
    'myresult/net1111TestLossACT_Train' + str(100 * float(temp_train) / num_train_instances)[
                                                                0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'test_loss': test_loss_act})
sio.savemat(
    'myresult/net1111TrainLossLOC_Train' + str(100 * float(temp_train) / num_train_instances)[
                                 0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'train_loss': train_loss_loc})
sio.savemat(
    'myresult/net1111TestLossLOC_Train' + str(100 * float(temp_train) / num_train_instances)[
                                 0:6] + 'Test' + str(
        100 * float(temp_test) / num_test_instances)[0:6] + '.mat', {'test_loss': test_loss_loc})

sio.savemat('myresult/net1111TrainAccuracyACT_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'train_acc': train_acc_act})
sio.savemat('myresult/net1111TestAccuracyACT_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'test_acc': test_acc_act})

#文件名包含了最终的训练和测试准确率信息。最后打印出测试准确率。
print(str(100 * float(temp_test) / num_test_instances)[0:6])

sio.savemat('myresult/net1111TrainAccuracyLOC_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'train_acc': train_acc_loc})
sio.savemat('myresult/net1111TestAccuracyLOC_Train' + str(
    100 * float(temp_train) / num_train_instances)[0:6] + 'Test' + str(100 * float(temp_test) / num_test_instances)[
                                                                   0:6] + '.mat', {'test_acc': test_acc_loc})