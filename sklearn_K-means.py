from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
data = np.genfromtxt('F:\我的学习资料\广东工业大学\网上学习课程\AL_MOOC\聚类\聚类算法\kmeans.txt', delimiter=' ')
plt.show()
# 设置k值
k = 4

# 训练模型
model = KMeans(n_clusters=k)
model.fit(data)


# 分类中心点坐标
centers = model.cluster_centers_
print("中心点坐标：\n", centers)

# 预测结果
result = model.predict(data)
print("数据所属的类：\n", result)

# model.labels_

# 画出各个数据点，用不同颜色表示分类
mark = ['or', 'ob', 'og', 'oy']
for i, d in enumerate(data):
    plt.plot(d[0], d[1], mark[result[i]])
# 画出各个分类的中心点
mark = ['*r', '*b', '*g', '*y']
for i, center in enumerate(centers):
    plt.plot(center[0], center[1], mark[i], markersize=20)
plt.show()
# 获取数据值所在的范围
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
# 等高线图
cs = plt.contourf(xx, yy, z)
# 显示结果
mark = ['or', 'ob', 'og', 'oy']
for i, d in enumerate(data):
    plt.plot(d[0], d[1], mark[result[i]])
# 画出各个分类的中心点
mark = ['*r', '*b', '*g', '*y']
for i, center in enumerate(centers):
    plt.plot(center[0], center[1], mark[i], markersize=20)
plt.show()
