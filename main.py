import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from datetime import datetime
from sklearn.model_selection import cross_val_score
import warnings

# 删除警告的
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
# 加载训练集
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
# 这部分代码提取了train_data1中的所有行，从第一列（索引0）之后的所有列。也就是说，它获取了所有特征数据，而忽略了第一列，因为第一列通常包含标签
train_data = train_data1.values[:,1:]
# 这部分代码提取了train_data1中的所有行，仅获取第一列（索引0）。这就是标签，它包含我们希望模型学习预测的目标
train_label = train_data1.values[:,0]
test_data1 = pd.read_csv('digit-recognizer/test.csv')
test_data = test_data1.values[:,0:]
print(train_data)


def showPic(data):
    plt.figure(figsize=(7, 7))
    # 查看前70幅图
    for digit_num in range(0, 70):
        # 7行10列
        plt.subplot(7, 10, digit_num + 1)
        # 将一维的数据重新转换成二维的28x28像素数组
        grid_data = data[digit_num].reshape(28, 28)
        #  这一行代码使用imshow函数显示图像。grid_data是刚刚转换得到的二维像素矩阵。interpolation="none"表示不进行插值，cmap="afmhot"表示使用热度图的颜色映射
        plt.imshow(grid_data, interpolation="none", cmap="afmhot")
        # 隐藏坐标轴
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 显示训练数据的图像
showPic(train_data)

# 创建PCA对象，指定要保留的主成分数量为22，对数据进行白化处理
pca = PCA(n_components=22, whiten=True)
# 对训练数据进行PCA降维
train_x = pca.fit_transform(train_data)
# 对测试数据进行相同的PCA变换
test_x = pca.transform(test_data)
# 打印训练数据在进行PCA之前和之后的形状
print(train_data.shape, train_x.shape)


def test(train_x, train_label):
    # 记录函数开始执行的时间
    start = datetime.now()
    # 创建一个SVM分类器，使用径向基核（RBF核）并设置C参数为10
    model = svm.SVC(kernel='rbf', C=10)
    # 使用交叉验证计算模型的准确率。这里采用了5折交叉验证，使用准确率作为评估指标
    metric = cross_val_score(model, train_x, train_label, cv=5, scoring='accuracy').mean()
    # 记录函数执行结束的时间
    end = datetime.now()
    # 打印交叉验证所花费的时间
    print('CV use: %f' % ((end - start).seconds))
    # 打印离线准确率，即使用训练数据评估模型的准确率
    print('Offline Accuracy is %f ' % (metric))
# 调用 test 函数，传入降维后的训练数据 train_x 和对应的标签 train_label，从而评估SVM模型的性能
test(train_x, train_label)


# 最终使用pca+svm
# 创建一个使用RBF核的SVM分类器，并设置C参数为10
SVM_model = svm.SVC(kernel='rbf', C=10)
# 创建一个PCA对象，指定要保留的主成分数量为22，并设置whiten=True进行白化处理
pca = PCA(n_components=22,whiten=True)
# 设置结果文件的名称为'PCA_SVM'
resultname = 'PCA_SVM'
# 使用PCA降维后的训练数据 train_x 和对应的标签 train_label 来训练SVM模型
SVM_model.fit(train_x,train_label)
# 使用训练好的模型对PCA降维后的测试数据 test_x 进行预测，结果保存在 test_y 中
test_y = SVM_model.predict(test_x)
# 创建一个包含测试数据预测结果的列表。其中，index + 1 表示图像的标识号，x 表示对应的预测标签
pred = [[index + 1, x] for index, x in enumerate(test_y)]
# 将测试结果保存为CSV文件。每一行包含图像标识号和对应的预测标签
np.savetxt(resultname+'.csv', pred, delimiter=',', fmt='%d,%d', header='ImageId,Label',comments='')
print('预测完成')
