import numpy as np
from random import shuffle
from collections import Counter
import pandas as pd
from random import shuffle
import cv2 as cv
# train_data_10000=[]
# number = 1
# for i in range(0,61):
#     train_name = 'F:/train_data/trian_data_0{}.npy'.format(number)
#     print(train_name)
#     train_data = np.load(train_name,allow_pickle=True)
#     for j in train_data:
#         picture =j[0]
#         picture =cv.resize(picture,(32,32))
#         target =j[1]
#         train_data_10000.append([picture,target])
#     #print(len(train_data_60000))
#     if len(train_data_10000) == 10000:
#         np.save('F:/re_train-data/train_data_10000_{}'.format(number),train_data_10000)
#         print('已保存10000张数据集')
#         del train_data_10000
#         train_data_10000 =[]
#         print(len(train_data_10000))
#     number = number + 1
#####################################数据平衡
# number =1
# WA,WD,SA,SD,W,A,S,D,NK =[],[],[],[],[],[],[],[],[]
# for i in range(1,7):
#     data = np.load('F:/re_train-data/train_data_10000_{}0.npy'.format(number),allow_pickle=True)
#     print('train_data_10000_{}0.npy'.format(number))
#     for split in data:
#         picture =split[0]
#         target = split[1]
#         if target == 0:
#             WA.append([picture,target])
#         elif target == 1:
#             WD.append([picture,target])
#         elif target == 2:
#             SA.append([picture,target])
#         elif target == 3:
#             SD.append([picture,target])
#         elif target == 4:
#             W.append([picture,target])
#         elif target == 5:
#             A.append([picture,target])
#         elif target == 6:
#             S.append([picture,target])
#         elif target == 7:
#             D.append([picture,target])
#         else:
#             NK.append([picture,target])
#     number =number +1
# shuffle(W)
# re_train_data_shunxuxiangjia_W =WA+WD+SA+SD+W[:6379]+A+S+D+NK
# print(len(re_train_data_shunxuxiangjia_W))
# np.save('F:/re_train-data/re_train_data_shunxuxiangjia_W',re_train_data_shunxuxiangjia_W)
# df = pd.DataFrame(re_train_data_shunxuxiangjia_W)
# print(df.head())
# print(Counter(df[1].apply(str)))
#######数据平衡2
train_data =np.load('F:/re_train-data/re_train_data_shunxuxiangjia_W.npy',allow_pickle=True)
shuffle(train_data) #打乱数据集
train_picture=[]
target=[]
for data in train_data:
    data_picture =data[0]
    data_target=data[1]
    train_picture.append(data_picture)
    target.append(data_target)
np.save('F:/re_train-data/train_data.npy',train_picture)
np.save('F:/re_train-data/target.npy',target)

