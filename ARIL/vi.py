import tkinter as tk
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import tkinter as tk
from tkinter import ttk
import torch
from torch.autograd import Variable
from tqdm import tqdm
#
# # def on_button_click():
# #     label.config(text="Hello, " + entry.get())
#
#
# def update_gui(progress, message):
#     progress_bar['value'] = progress
#     result_text.insert(tk.END, message + '\n')
#     window.update()
#
# def start_model():
#     # 设置batch_size为512,表示每批次处理512个样本。
#     batch_size = 600
#
#     # 从'mydata/test_data.mat'文件中读取测试数据和标签。
#     # test_data_amp是测试数据,test_activity_label和test_location_label分别是activity和location的标签。
#     # 将两个标签在第1维(列)上拼接得到完整的标签test_label。num_test_instances是测试样本总数。
#     data_amp = sio.loadmat('data_for_test/extracted_data.mat')
#     test_data_amp = data_amp['test_data_amp']
#     test_data = test_data_amp
#
#     test_activity_label = data_amp['test_activity_label']
#     test_location_label = data_amp['test_location_label']
#     test_label = np.concatenate((test_activity_label, test_location_label), 1)
#
#     num_test_instances = len(test_data)
#
#     # 将numpy格式的test_data和test_label转换为PyTorch的Tensor,并指定数据类型。然后用TensorDataset将数据和标签封装成数据集,
#     # 再用DataLoader创建数据加载器,设置batch大小为512,shuffle为False表示不打乱数据顺序。
#     test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
#     test_label = torch.from_numpy(test_label).type(torch.LongTensor)
#     # test_data = test_data.view(num_test_instances, 1, -1)
#     # test_label = test_label.view(num_test_instances, 2)
#
#     test_dataset = TensorDataset(test_data, test_label)
#     test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#
#     # 加载预训练的模型aplnet,将其移到GPU上,并设置为评估模式。
#     # aplnet = torch.load('weights/net1111_Train100.0Test88.129Train99.910Test95.683.pkl')
#     aplnet = torch.load('myweights/net1111epoch107Train92.291Test85.0Train100.0Test100.0.pkl')
#     aplnet = aplnet.cuda().eval()
#
#     # 初始化location和activity的正确预测数为0。
#     correct_test_loc = 0
#     correct_test_act = 0
#
#     # 开始遍历测试数据,每次取一个batch的数据。
#     # 将数据移到GPU上,并转换为Variable。将标签拆分为activity标签和location标签,也移到GPU上。
#     # with torch.no_grad()表示不追踪梯度,因为测试时不需要更新参数。
#     for i, (samples, labels) in enumerate(test_data_loader):
#         with torch.no_grad():
#             samplesV = Variable(samples.cuda())
#             labels_act = labels[:, 0].squeeze()
#             labels_loc = labels[:, 1].squeeze()
#             labelsV_act = Variable(labels_act.cuda())
#             labelsV_loc = Variable(labels_loc.cuda())
#
#             progress = (i + 1) / len(test_data_loader) * 100
#             message = f"Batch {i + 1}/{len(test_data_loader)}, Activity Accuracy: {correct_test_act.cpu().numpy() / num_test_instances:.2f}, Location Accuracy: {correct_test_loc.cpu().numpy() / num_test_instances:.2f}"
#             update_gui(progress, message)
#             # 打印labelsV_act的数据
#             # print("labelsV_act:")
#             # # print(labelsV_act)
#             # print(labels_act.cpu().numpy())
#             # print(type(labels_act.cpu().numpy()))
#             # print(str(labels_act.cpu().numpy()))
#             # print(type(str(labels_act.cpu().numpy())))
#             # print()
#
#             # 将一个batch的数据输入到模型aplnet中, 得到预测的activity标签predict_label_act和location标签predict_label_loc,
#             # 其他返回值是中间特征图, 这里不使用。
#             predict_label_act, predict_label_loc, _, _, _, _, _, _, _ = aplnet(samplesV)
#
#             # 对于activity预测结果,取概率最大的类别作为预测标签,保存到'vis/actResult.mat'文件中。
#             # 将预测标签与真实标签比较,统计预测正确的样本数,并打印当前的准确率。
#             prediction = predict_label_act.data.max(1)[1]
#             # print("prediction:")
#             # print(prediction.cpu().numpy())
#             # print(type(prediction.cpu().numpy()))
#             # print(str(prediction.cpu().numpy()))
#             # print(type(str(prediction.cpu().numpy())))
#
#
#     pre_activity = ""
#     if str(prediction.cpu().numpy()) == '[0]':
#         pre_activity = "dow"
#     elif str(prediction.cpu().numpy()) == '[1]':
#         pre_activity = "up"
#     elif str(prediction.cpu().numpy()) == '[2]':
#         pre_activity = "circle"
#     elif str(prediction.cpu().numpy()) == '[3]':
#         pre_activity = "clap"
#     elif str(prediction.cpu().numpy()) == '[4]':
#         pre_activity = "cross"
#     else :
#         pre_activity = "tick"
#
#     label.config(text="检测到" + pre_activity + "动作")
#
#
#
# # 创建主窗口
# window = tk.Tk()
# window.title("Interactive UI Example")
#
# # 创建标签、输入框和按钮
# label = tk.Label(window, text="Welcome to the Wi-Fi sensing human behavior recognition system")
# label.pack()
#
# progress_bar = ttk.Progressbar(window, length=300, mode='determinate')
# progress_bar.pack()
# # entry = tk.Entry(window)
# # entry.pack()
#
# # button = tk.Button(window, text="Click Me", command=on_button_click)
# # button.pack()
# result_text = tk.Text(window, height=10, width=50)
# result_text.pack()
#
# button = tk.Button(window, text="start model", command=start_model)
# button.pack()
#
# # 运行主循环
# window.mainloop()
#


import tkinter as tk
from tkinter import ttk
import time
import threading



# 定义更新GUI的函数
def update_gui(progress, message):
    progress_bar['value'] = progress
    result_text.insert(tk.END, message + '\n')
    window.update()

# 定义测试函数
def run_test():
    total_time = 1  # 设置总时间为10秒
    start_time = time.time()

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        progress = (elapsed_time / total_time) * 100

        if progress >= 100:
            progress = 100
            message = "测试完成!"
            update_gui(progress, message)
            break

        message = f"已测试时间: {elapsed_time:.2f}秒"
        update_gui(progress, message)
        time.sleep(0.1)  # 添加短暂延迟以模拟测试过程


pre_activity = ""
def start_model():
    start_time = time.time()
    total_time = 3

    update_gui((time.time()-start_time)/total_time*100,f"已测试时间: {time.time()-start_time:.2f}秒")
    # 设置batch_size为,表示每批次处理100个样本。
    batch_size = 1

    data_amp = sio.loadmat('data_for_test/extracted_data.mat')
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



    aplnet = torch.load('myweights/net1111epoch107Train92.291Test85.0Train100.0Test100.0.pkl')
    aplnet = aplnet.cuda().eval()


    # 开始遍历测试数据,每次取一个batch的数据。
    # 将数据移到GPU上,并转换为Variable。将标签拆分为activity标签和location标签,也移到GPU上。
    # with torch.no_grad()表示不追踪梯度,因为测试时不需要更新参数。
    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            update_gui((time.time()-start_time)/total_time*100,f"已测试时间: {time.time()-start_time:.2f}秒")

            samplesV = Variable(samples.cuda())
            labels_act = labels[:, 0].squeeze()
            labels_loc = labels[:, 1].squeeze()
            labelsV_act = Variable(labels_act.cuda())
            labelsV_loc = Variable(labels_loc.cuda())

            # 将一个batch的数据输入到模型aplnet中, 得到预测的activity标签predict_label_act和location标签predict_label_loc,
            # 其他返回值是中间特征图, 这里不使用。
            predict_label_act, predict_label_loc, _, _, _, _, _, _, _ = aplnet(samplesV)

            # 对于activity预测结果,取概率最大的类别作为预测标签,保存到'vis/actResult.mat'文件中。
            # 将预测标签与真实标签比较,统计预测正确的样本数,并打印当前的准确率。
            prediction = predict_label_act.data.max(1)[1]


    if str(prediction.cpu().numpy()) == '[0]':
        pre_activity = "down"
    elif str(prediction.cpu().numpy()) == '[1]':
        pre_activity = "up"
    elif str(prediction.cpu().numpy()) == '[2]':
        pre_activity = "circle"
    elif str(prediction.cpu().numpy()) == '[3]':
        pre_activity = "clap"
    elif str(prediction.cpu().numpy()) == '[4]':
        pre_activity = "cross"
    else :
        pre_activity = "tick"

    update_gui((time.time() - start_time) / total_time * 100, f"已测试时间: {time.time() - start_time:.2f}秒")

    label1.config(text="检测到" + pre_activity + "动作")




def start_tests():
    # thread1 = threading.Thread(target=run_test())
    # thread2 = threading.Thread(target=start_model())
    #
    # thread2.start()
    # thread1.start()
    start_model();
    run_test();
    label1.config(text="检测到" + pre_activity + "动作")


# 创建Tkinter窗口
window = tk.Tk()
window.title("Wi-Fi Sense")

# 创建标签和进度条
label = tk.Label(window, text="Welcome to the Wi-Fi sensing human behavior recognition system")
label.pack()

label1 = tk.Label(window, text="")
label1.pack()

progress_bar = ttk.Progressbar(window, length=500, mode='determinate')
progress_bar.pack()

# 创建文本框用于显示测试结果
result_text = tk.Text(window, height=20, width=50)
result_text.pack()


# 创建开始测试的按钮
start_button = tk.Button(window, text="开始测试", command=start_model)
start_button.pack()

# 运行Tkinter事件循环
window.mainloop()