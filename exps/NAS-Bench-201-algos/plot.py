# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 21:16
# @Author  : LIU YI




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import re
import ast

file1 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-61-T-14-Sep-at-19-06-38.log'
file2 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-61-T-14-Sep-at-19-08-07.log'
file3 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-61-T-14-Sep-at-19-11-04.log'
file4 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-61-T-14-Sep-at-19-13-13.log'
file5 = 'output_search-cell-nas-bench-201_GDAS-cifar10-BN1_seed-61-T-14-Sep-at-20-33-15.log'
file6 = 'output_search-cell-nas-bench-201_GDAS-cifar10-BN1_seed-61-T-14-Sep-at-20-34-48.log'
file7 = 'output_search-cell-nas-bench-201_GDAS-cifar10-BN1_seed-61-T-14-Sep-at-20-36-25.log'
file8 = 'output_search-cell-nas-bench-201_GDAS-cifar10-BN1_seed-61-T-14-Sep-at-20-37-52.log'
file9 = 'output_search-cell-nas-bench-201_GDAS-cifar10-BN1_seed-610915-T-14-Sep-at-09-50-01.log'
file10 = 'output_search-cell-nas-bench-201_GDAS-cifar10-BN1_seed-610915-T-14-Sep-at-09-51-36.log'
file11 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-610915-T-14-Sep-at-09-46-41.log'
file12 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-610915-T-14-Sep-at-09-48-42.log'
file13 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-61-T-13-Sep-at-08-10-48.log'    # Leaf-0001
file14 = 'output_search-cell-nas-bench-201_GDAS-cifar100-BN1_seed-61-T-13-Sep-at-08-12-23.log'    # joey-4T4


file_proposal = 'FedNAS_Search_darts.log'
file_proposal1 = 'Ours_Search_darts.log'
file_proposal2 = 'FedNAS_128.log'

def before():


    tep = pd.DataFrame()
    files = [file1, file2, file3, file4]
    cifar10_files = [file7, file8]

    files610915 = [file9, file10, file11, file12]
    # files = files610915
    files = cifar10_files

    files = [file13, file14]





    # names = ['Personalize Arch+DL', 'DL only', 'only Personalized Arch', 'FL']
    # names = ['cifar10_ousr', 'cifar10_baseline', 'cifar100_ours', 'cifar100_baseline']
    names = ['pFed_NAS', 'baseline']
    for file in files:
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

    tep.columns = names

    a = tep.plot()
    plt.tick_params(labelsize = 15)
    plt.xlabel("Training Rounds", size = 15)
    plt.ylabel("Mean Accuracy", size = 15)
    plt.grid(linestyle = '-.')
    plt.legend(prop = {'size':12})
    plt.savefig('Figure3.eps', dpi = 600, format = 'eps')
    plt.show()


    tep = pd.DataFrame()

    for file in files:
        result = []
        for user in range(5):
            result.append([])

        for line in open(file):
            for user in range(5):
                a = re.search('user {}'.format(user), line)
                if a:
                    if '||||' in line:
                        result[user].append(float(re.findall('valid_top1=(.+?)%',line)[0]))

        result = pd.DataFrame(result)
        result = result.T
        result.columns = ['user0', 'user1', 'user2', 'user3', 'user4']
        result['avg'] = result.mean(axis = 1)
        tep[file] = result['avg']

    tep.columns = names

    tep.plot()
    plt.show()


#
# genotype_list = {}
# user_list = {}
# user = 0
# for line in open(file_proposal2):
#     if "<<<--->>>" in line:
#         tep_dict = ast.literal_eval(re.search('({.+})', line).group(0))
#         count = 0
#         for j in tep_dict['normal']:
#             for k in j:
#                 if 'skip_connect' in k[0]:
#                     count += 1
#         if count == 2:
#             genotype_list[user%5] = tep_dict
#             user_list[user%5] = user/5
#         user+=1
#
# # print(genotype_list)
# skip_count = {}
#
# for one in range(5):
#     skip_count[one] = []
#
# for one in range(5):
#     for i in genotype_list[one]:
#         count = 0
#         for j in i['normal']:
#             for k in j:
#                 if 'skip_connect' in k[0]:
#                     count += 1
#         skip_count[one].append(count)
#
#
#
# file3 = 'cifar100_PFL.log'


#python 画柱状图折线图
#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
a=[87.79999707,97.44999829,97.24999756,97.74999744,98.09999768]  #数据
b=[91.7,95.75,95.05,95.25,96.45]
c=[0.978526,1.716076,1.254538,1.23553,1.421056]
d=[1.12603,1.12603,1.12603,1.12603,1.12603]
l=[i for i in range(5)]
plt.figure(figsize=(10,10))
width = 0.3
n = 2
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

# fmt='%.2f%%'
# yticks = mtick.FormatStrFormatter(fmt)  #设置百分比形式的坐标轴
lx=['User0','User1','User2','User3','User4']

fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in range(len(l)):
        l[i] = l[i] - width
plt.bar(l, a, width = width,label='pFed_NAS')
for i in range(len(l)):
        l[i] = l[i] + width

plt.bar(l,b,width = width,label='baseline')

for i in range(len(l)):
        l[i] = l[i] - width/2

# ax1.plot(l, a,'or-',label='Model Performance')
# ax1.yaxis.set_major_formatter(yticks)
# for i,(_x,_y) in enumerate(zip(l,b)):
#     plt.text(_x,_y,b[i],color='black',fontsize=10,)  #将数值显示在图形上
ax1.legend(loc='upper left')
ax1.set_ylim([82, 105]);
plt.ylabel("Model Performance (%)")
# plt.legend(prop={'family':'SimHei','size':8})  #设置中文
ax2 = ax1.twinx() # this is the important function
# plt.bar(l,c,alpha=0.3,color='blue',label=u'产量')
for i in range(len(l)):
        l[i] = l[i] - width/2
ax2.plot(l, c,label='pFed_NAS')
for i in range(len(l)):
        l[i] = l[i] + width
ax2.plot(l, d,label='baseline')

ax2.legend(loc=0)
ax2.set_ylim([0.4, 2])  #设置y轴取值范围
plt.ylabel("Model Param Size (Mb)")
# plt.legend(prop={'family':'SimHei','size':8},loc="upper left")
plt.xticks(l,lx)
plt.savefig('Exp.eps', dpi = 600, format = 'eps')
plt.show()




