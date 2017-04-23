import numpy as np
import pandas as pd
import operator
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool
import time


class HandWritingTestByKNN:

    # 通过余弦值判断两个向量的相似度
    def knn_cos(self, train_x, train_y, test_x):
        """
        归一化处理
        linalg.norm(x), return sum(abs(xi)**2)**0.5
        """
        norms_train = np.apply_along_axis(np.linalg.norm, 1, train_x) + 1.0e-7
        norms_test = np.apply_along_axis(np.linalg.norm, 1, test_x) + 1.0e-7
        train_x = train_x / np.expand_dims(norms_train, -1)
        test_x = test_x / np.expand_dims(norms_test, -1)

        # cosine
        corr = np.dot(test_x, np.transpose(train_x))
        argmax = np.argmax(corr, axis=1)
        preds = train_y[argmax]
        return preds

    # 通过欧氏距离判断两个向量的相似度
    def classify(self, test_data, train_data, labels, k):
        data_size = train_data.shape[0]
        diff_mat = np.tile(test_data, (data_size, 1)) - train_data
        sq_diff_mate = diff_mat ** 2
        distance = sq_diff_mate.sum(axis=1) ** 0.5
        # distance = np.linalg.norm(diff_mat, axis=1)
        sorted_dist_indicies = np.argsort(distance)
        class_count = {}
        for i in range(k):
            vote_label = labels[sorted_dist_indicies[i]]
            class_count[vote_label] = class_count.get(vote_label, 0) + 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    # 通过欧氏距离判断两个向量的相似度
    def knn_distance(self, train_data, train_label, test_data):
        preds = np.empty(test_data.shape[0])    # 初始化test_data的预测结果
        for index, data in enumerate(test_data):
            test_result_label = self.classify(data, train_data, train_label, 3)
            preds[index] = test_result_label
        return preds

    # scikit-learn提供的knn算法
    def knn_sklearn(self, train_data, train_label, test_data):
        pca = PCA(n_components=0.8)
        train_x = pca.fit_transform(train_data)
        test_x = pca.transform(test_data)
        # knn regression
        neighbor = KNeighborsClassifier(n_neighbors=4)
        neighbor.fit(train_x, train_label)
        preds = neighbor.predict(test_x)
        return preds

    # 判断预测结果
    def validate(self, func_name, preds, test_y):
        count = len(preds)
        correct = (preds == test_y).sum()
        acc = float(correct) / count
        print("%s正确率: %f" %(func_name, acc))
        return acc

    # 读取数据
    def data_prepare(self):
        TRAIN_NUM = 2200
        TEST_NUM = 4200
        data = pd.read_csv('train.csv')
        train_data = data.values[0:TRAIN_NUM, 1:]
        train_label = data.values[0:TRAIN_NUM, 0]
        test_data = data.values[TRAIN_NUM:TEST_NUM, 1:]
        test_label = data.values[TRAIN_NUM:TEST_NUM, 0]
        return (train_data, train_label, test_data, test_label)

    def start(self, func_name='knn_cos'):
        print('start ',func_name)
        func = getattr(self, func_name)
        start = time.time()
        data = self.data_prepare()
        preds = func(*data[:3])
        acc = self.validate(func_name, preds, data[3])
        print("run time: %.2fs" %(time.time() - start))


if __name__ == '__main__':
    hw = HandWritingTestByKNN()
    func_list = ['knn_distance', 'knn_cos', 'knn_sklearn']
    pool = Pool()
    pool.map(hw.start, func_list)
    pool.close()
    pool.join()