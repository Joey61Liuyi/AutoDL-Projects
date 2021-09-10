# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 21:16
# @Author  : LIU YI




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


import re

file1 = 'cifar100_individual.log'
file2 = 'cifar100_FL.log'
file3 = 'cifar100_PFL.log'


tep = pd.DataFrame()


for file in [file1, file2, file3]:
    result = []
    for user in range(5):
        result.append([])

    for line in open(file):
        for user in range(5):
            a = re.search('^User {}'.format(user), line)
            if a:
                if 'evaluate' in line:
                    result[user].append(float(re.findall('accuracy@1=(.+?)%',line)[0]))

    result = pd.DataFrame(result)
    result = result.T
    result.columns = ['user0', 'user1', 'user2', 'user3', 'user4']
    result['avg'] = result.mean(axis = 1)
    tep[file] = result['avg']

tep.columns = ['without aggregation', 'Traditional FL', 'Personalize Arch']

tep.plot()
plt.show()



# 绘图参数全家桶
# params = {
#     'axes.labelsize': '13',
#     'xtick.labelsize': '12',
#     'ytick.labelsize': '12',
#     'legend.fontsize': '13',
#     'figure.figsize': '4, 3',
#     'figure.dpi':'300',
#     'figure.subplot.left':'0.165',
#     'figure.subplot.right':'0.965',
#     'figure.subplot.bottom':'0.135',
#     'figure.subplot.top':'0.925',
#     'pdf.fonttype':'42',
#     'ps.fonttype':'42',
# }
# pylab.rcParams.update(params)
#
# # data = pd.read_excel('E:\香港\dissertation\画图图标/TA_NAS Experiment Record.xlsx',names=['Teacher', 'T-ACC', 'S-lenet'], sheet_name='Cifar10', header=None, usecols=[14, 15, 20], skiprows=range(0, 32), skipfooter=6)
# labels = ['2', '3', '4', '5', '6', '7', '8']
# # T_ACC = [96.05, 95.87, 94.89, 93.78, 94.20, 94.46]
# S_lenet = [71.73, 69.48, 69.94, 70.64, 70.19, 68.80, 70.12]
# # labels.reverse()
# # T_ACC.reverse()
# # S_lenet.reverse()
# # print(T_ACC)
#
# # 设置柱形的间隔
# width = 0.3  # 柱形的宽度
#
# x = 1 * np.arange(7)
#
# f, ax1 = plt.subplots()
# # 设置左侧Y轴对应的figure
# # ax1.set_ylabel('T-ACC', color='red')
# # ax1.set_ylim(93, 98)
# # ax1.bar(x - width / 2, T_ACC, width=width, color='red', label='T_ACC')
# # ax1.plot(x - width / 2 , T_ACC, color='red')
#
# # 设置右侧Y轴对应的figure
# # ax2 = ax1.twinx()
# ax1.set_ylabel('Student Accuracy')
# ax1.set_xlabel('widen_times')
# ax1.set_ylim(68, 72)
# ax1.bar(x + width / 2, S_lenet, width=width, color='tab:blue')
# # ax2.plot(x + width / 2, S_lenet, color='tab:blue')
#
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels)
#
#
# # x = 0.1 * np.arange(30)
# # y = 3 * x + 83
# # ax1.plot(x - width / 2 , y, color='tab:red', linestyle='--')
# #
# # y = - 0.65 * (x - 1.3) * (x - 1.3) + 71
# # ax2.plot(x + width / 2 , y, color='tab:blue', linestyle='--')
#
# plt.tight_layout()
# # plt.savefig("similarity.png")
# plt.show()