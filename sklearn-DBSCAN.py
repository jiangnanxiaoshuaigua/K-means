'''
DBSCAN = Density - Based Spatial Clustering of Applications with Noise
该算法将具有足够高密度的区域划分为簇，并可以发现任何形状的聚类
DBSCAN和K-Means的比较：
    DBSCAN不需要输入聚类个数
    聚类簇的形状没有要求
    可以在需要时输入过滤噪声的参数
'''
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# 载入数据
data = np.genfromtxt('F:\我的学习资料\广东工业大学\网上学习课程\AL_MOOC\聚类\聚类算法\kmeans.txt',delimiter=' ')

# 训练模型
# eps距离阈值，min_sample核心对象在eps领域的样本数阈值
model = DBSCAN(eps=1.5, min_samples=4)
model.fit(data)
result = model.fit_predict(data)
print(result)
mark = ['or', 'ob', 'og', 'oy', 'ok', 'om']
for i, d in enumerate(data):
    plt.plot(d[0], d[1], mark[result[i]])
plt.show()