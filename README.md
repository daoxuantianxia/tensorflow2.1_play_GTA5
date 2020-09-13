# tensorflow2.1_play_GTA5
using the tensorflow 2.1 to playing GTA5
#环境要求：tensorflow 2.1.0，pillow, opencv, numpy, pywin32
#注释：自制数据集请使用grab_train_data这个文件（收集图片的尺寸为32*32），getkeys是获取键盘操作的代码（自制数据集要用），directkeys是控制键盘的代码（人机操控键盘的代码）；可以参考balance_03的方法平衡数据集，因为开车的时候，W（向前）的数据一定是最多的，尽量平衡它与其他数据，不然训练出来的模型大概率是无脑向前冲！！ LeNet_train是训练神经网络的代码（已经加了回调函数），LeNet_test是测试代码（用它就可以人机玩GTA5啦）；剩下的文件只有神经网络的结构不同，其余的与LeNet一致；
