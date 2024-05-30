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
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
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

#设置batch_size为512,表示每批次处理512个样本。
batch_size = 600



#从'mydata/test_data.mat'文件中读取测试数据和标签。
#test_data_amp是测试数据,test_activity_label和test_location_label分别是activity和location的标签。
#将两个标签在第1维(列)上拼接得到完整的标签test_label。num_test_instances是测试样本总数。
data_amp = sio.loadmat('mydata/test_data_Room1+40cm.mat')
test_data_amp = data_amp['test_data_amp']
test_data = test_data_amp

test_activity_label = data_amp['test_activity_label']
test_location_label = data_amp['test_location_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)



#将numpy格式的test_data和test_label转换为PyTorch的Tensor,并指定数据类型。然后用TensorDataset将数据和标签封装成数据集,
#再用DataLoader创建数据加载器,设置batch大小为512,shuffle为False表示不打乱数据顺序。
test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#加载预训练的模型aplnet,将其移到GPU上,并设置为评估模式。
#aplnet = torch.load('weights/net1111_Train100.0Test88.129Train99.910Test95.683.pkl')
aplnet = torch.load('myweights/net1111epoch106Train84.133Test79.047Train100.0Test100.0.pkl')
aplnet = aplnet.cuda().eval()

# 初始化location和activity的正确预测数为0。
correct_test_loc = 0
correct_test_act = 0



# 开始遍历测试数据,每次取一个batch的数据。
# 将数据移到GPU上,并转换为Variable。将标签拆分为activity标签和location标签,也移到GPU上。
# with torch.no_grad()表示不追踪梯度,因为测试时不需要更新参数。
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act.cuda())
        labelsV_loc = Variable(labels_loc.cuda())

        # 打印labelsV_act的数据
        print("labelsV_act:")
        print(labelsV_act)
        sio.savemat('vis/actAssign.mat', {'act_true': labels_act.cpu().numpy()})

        # 将一个batch的数据输入到模型aplnet中, 得到预测的activity标签predict_label_act和location标签predict_label_loc,
        # 其他返回值是中间特征图, 这里不使用。
        predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)

        # for tsne visualization
        # 这部分代码是为了可视化中间特征图, 将模型的各层输出保存到mat文件中,是为了后续的t - SNE可视化。
        # 将输出的tensor展平成(batch_size, feature_size)的形状,
        # 然后转为numpy数组, 最后保存到mat文件。
        # act1, loc1, x, c1, c2, c3, c4, act, loc = aplnet(samplesV)
        # sio.savemat('vis/fig_tsne/out_act_conf.mat', {'out_max': act1.view(act1.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_loc_conf.mat', {'out_max': loc1.view(loc1.shape[0]
        # , -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_maxpool.mat', {'out_max': x.view(x.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c1.mat', {'out_max': c1.view(c1.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c2.mat', {'out_max': c2.view(c2.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c3.mat', {'out_max': c3.view(c3.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c4.mat', {'out_max': c4.view(c4.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_fc_act.mat', {'out_max': act.view(act.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_fc_loc.mat', {'out_max': loc.view(loc.shape[0], -1).cpu().numpy()})


        #对于activity预测结果,取概率最大的类别作为预测标签,保存到'vis/actResult.mat'文件中。
        # 将预测标签与真实标签比较,统计预测正确的样本数,并打印当前的准确率。
        prediction = predict_label_act.data.max(1)[1]
        print("prediction:")
        print(prediction)
        sio.savemat('vis/actResult.mat',{'act_prediction':prediction.cpu().numpy()})
        correct_test_act += prediction.eq(labelsV_act.data.long()).sum()
        print(correct_test_act.cpu().numpy()/num_test_instances)


        #同样地,对location预测结果进行处理,保存结果,统计正确样本数,打印准确率。
        # prediction = predict_label_loc.data.max(1)[1]
        # sio.savemat('vis/locResult.mat', {'loc_prediction': prediction.cpu().numpy()})
        # correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()
        # print(correct_test_loc.cpu().numpy() / num_test_instances)

