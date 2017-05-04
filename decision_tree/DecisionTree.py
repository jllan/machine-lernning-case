from math import log
import operator

# 构造数据集
def createDataSet():
    dataSet = [
        [1, 1, 0, 1, 'no'],
        [1, 0, 2, 0, 'no'],
        [1, 1, 1, 1, 'yes'],
        [0, 1, 2, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 0, 0, 'no'],
        [1, 1, 2, 0, 'yes'],
        [1, 1, 1, 0, 'no'],
        [0, 1, 1, 1, 'no'],
        [1, 0, 1, 1, 'no'],
    ]
    labels = ['年龄','长相','收入','公务员']
    return dataSet, labels


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 统计数据集中每个分类的数量
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    # 计算每个分类的熵，然后就和得到整个数据集的熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 每个分类的数量/总数据集的数量=每个分类的概率
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


# 划分数据集，根据每个特征的不同值把整个数据集划分为多个数据子集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 用第i个特征的一个value划分一次数据集得到一个数据子集，计算该数据子集的熵
# 对每个value划分得到的数据子集的熵就和得到用该特征划分数据集后的熵
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 最后一列是标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # 遍历每个特征
        featList = [example[i] for example in dataSet]  # 所有数据的第i个特征
        uniqueVals = set(featList)  # 对特征值去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 用第i个特征的一个value来划分数据集
            prob = len(subDataSet) / float(len(dataSet))  # 第i个特征等于value的个数/整个数据集的数量
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 用第i个特征进行数据划分后的信息增益
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature


# 返回类别中数量最多的那个分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别完全相同则停止划分
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 根据决策树来构建分类器
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)



if __name__ == '__main__':
    dataSet, labels = createDataSet()
    tree = createTree(dataSet, labels)
    storeTree(tree, '决策树.txt')
    # dt = grabTree('决策树.txt')
    print(tree)
    # 测试
    labels = ['年龄', '长相', '收入', '公务员']
    label = classify(tree, labels, [1, 0, 1, 1])
    print(label)