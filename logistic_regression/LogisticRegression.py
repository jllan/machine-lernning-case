import numpy as np
import matplotlib.pyplot as plt
import time


def load_data():
    train_x = []
    train_y = []
    with open('testSet.txt') as f:
        for line in f.readlines():
            line = line.strip().split()
            train_x.append([1.0, float(line[0]), float(line[1])])
            train_y.append(float(line[2]))
    return np.mat(train_x), np.mat(train_y).transpose()


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#        train_y is mat datatype too, each row is the corresponding label
#        opts is optimize option include step and maximum number of iterations
def train(train_x, train_y, opts):
    start_time = time.time()
    num_samples, num_features = np.shape(train_x)
    alpha = opts['alpha']
    max_iter = opts['max_iter']
    weights = np.ones((num_features, 1))

    for k in range(max_iter):
        if opts['optimize_type'] == 'grad_descent':  # gradient descent algorilthm
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error
        elif opts['optimize_type'] == 'stoc_grad_descent':  # stochastic gradient descent
            for i in range(num_samples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error
        elif opts['optimize_type'] == 'smooth_stoc_grad_descent':  # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            data_index = list(range(num_samples))
            for i in range(num_samples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                rand_index = int(np.random.uniform(0, len(data_index)))
                output = sigmoid(train_x[rand_index, :] * weights)
                error = train_y[rand_index, 0] - output
                weights = weights + alpha * train_x[rand_index, :].transpose() * error
                del(data_index[rand_index])  # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')

    print('训练结束，花费时间： %fs!' % (time.time() - start_time))
    return weights


# test your trained Logistic Regression model given test set
def test(weights, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def show(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    num_samples, num_features = np.shape(train_x)
    if num_features != 3:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    # 画出所有的点
    for i in range(num_samples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # 画出分割线
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    #　加载数据
    print("step 1: load data...")
    train_x, train_y = load_data()
    test_x = train_x
    test_y = train_y

    #　训练
    print("step 2: training...")
    opts = {'alpha': 0.01, 'max_iter': 20, 'optimize_type': 'smooth_stoc_grad_descent'}
    optimalWeights = train(train_x, train_y, opts)

    #　测试
    print("step 3: testing...")
    accuracy = test(optimalWeights, test_x, test_y)

    #　查看结果
    print("step 4: show the result...")
    print('The classify accuracy is: %.3f%%' % (accuracy * 100))
    show(optimalWeights, train_x, train_y)
