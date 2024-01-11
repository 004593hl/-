import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from datetime import datetime
from sklearn.model_selection import cross_val_score
import warnings

#删除警告的

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

train = pd.read_csv('digit-recognizer/train.csv')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)


# Train_data 中存储了训练集的784个特征，Test_data存储了测试集的784个特征，train_lable则存储了训练集的标签
# 可以看出这道题是典型的监督学习问题
train_data1 = pd.read_csv('digit-recognizer/train.csv')
train_data = train_data1.values[:,1:]
train_label = train_data1.values[:,0]
test_data1 = pd.read_csv('digit-recognizer/test.csv')
test_data = test_data1.values[:,0:]
print(train_data)


def showPic(data):
    plt.figure(figsize=(7, 7))
    # 查看前70幅图
    for digit_num in range(0, 70):
        plt.subplot(7, 10, digit_num + 1)
        grid_data = data[digit_num].reshape(28, 28)  # reshape from 1d to 2d pixel array
        plt.imshow(grid_data, interpolation="none", cmap="afmhot")
        plt.xticks([])
        plt.yticks([])
    # plt.tight_layout()
    plt.show()


showPic(train_data)


# 初始数据有784个维度，现在需要对其降维处理，使用主成分分析法

'''
def getcomponent(inputdata):
    pca = PCA()
    pca.fit(inputdata)
    # 累计贡献率 又名 累计方差贡献率 不要简单理解为 解释方差！！！
    EV_List = pca.explained_variance_
    EVR_List = []
    for j in range(len(EV_List)):
        EVR_List.append(EV_List[j] / EV_List[0])
    for j in range(len(EVR_List)):
        if (EVR_List[j] < 0.10):
            print('Recommend %d:' % j)
            return j


getcomponent(train_data)

'''
pca = PCA(n_components=22, whiten=True)
train_x = pca.fit_transform(train_data)
test_x = pca.transform(test_data)  # 数据转换
print(train_data.shape, train_x.shape)


def test(train_x, train_label):
    start = datetime.now()
    model = svm.SVC(kernel='rbf', C=10)
    metric = cross_val_score(model, train_x, train_label, cv=5, scoring='accuracy').mean()
    end = datetime.now()
    print('CV use: %f' % ((end - start).seconds))
    print('Offline Accuracy is %f ' % (metric))

test(train_x, train_label)


# 最终使用pca+svm
SVM_model = svm.SVC(kernel='rbf', C=10)
pca = PCA(n_components=22,whiten=True)
resultname = 'PCA_SVM'
# modeltest(train_x,train_label,SVM_model)
SVM_model.fit(train_x,train_label)
test_y = SVM_model.predict(test_x)
pred = [[index + 1, x] for index, x in enumerate(test_y)]
np.savetxt(resultname+'.csv', pred, delimiter=',', fmt='%d,%d', header='ImageId,Label',comments='')
print('预测完成')
