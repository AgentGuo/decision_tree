#!/usr/bin/env python
# coding: utf-8

# # 决策树

# ## 数据加载

# In[1]:


import csv
import random
from copy import deepcopy


# ![U1yZXd.png](https://s1.ax1x.com/2020/07/12/U1yZXd.png)

# In[2]:


def loadTrainData(filename):
    data = []
    f = list(csv.reader(open(filename, 'r')))[1:]   # 读取去掉表头的部分
    embarkedDist = {'C':1, 'Q':2, 'S':3}    # 无缺失值时，'C':1, 'Q':2, 'S':3
    sibspParch = [1, 2]
    for line in f:
        if int(line[6]) == 0:  # 转化sibsp
            sibsp = 0
        elif int(line[6]) <= 2:
            sibsp = 1
        else:
            sibsp = 2
        
        if int(line[7]) == 0:   # 转化sibsp
            parch = 0
        elif int(line[7]) <= 2:
            parch = 1
        else:
            parch = 2
        dataDist={'survived':int(line[1]),
             'pclass': int(line[2]) - 1,
             'sex': 0 if line[4] == 'male' else 1,   # male:0  female:1
             'age': 0 if len(line[5]) == 0 else float(line[5]),   # 有缺失值保存为0
             'sibsp': sibsp,
             'parch': parch,
             'fare': float(line[9]),
             'cabin': 0 if len(line[10]) == 0 else 1,   # 有缺失值：0  无缺失值：1
             'embarked': 0 if len(line[11]) == 0 else embarkedDist[line[11]],   # 有缺失值：0  无缺失值保存为1、2、3
                  'w': 1}  # 初始化权重
        data.append(dataDist)
    random.shuffle(data)     # 将加载好的数据打乱
    trainData = data[:741]  # 按大致5:1划分训练集和验证集
    devData = data[741:]
    return trainData, devData


# In[3]:


trainData, devData = loadTrainData('train.csv')
print(len(trainData), len(devData), trainData[0])


# ## 构建决策树

# ### 1. 结点定义

# In[4]:


class Node:
    def __init__(self,attribute):
        self.son = []  # 结点的孩子
        self.attribute = attribute    # 结点当前的划分属性
        self.boundary = -1  # 当前结点划分属性为连续值时才修改该属性
        self.kind = -1    # 当前结点的种类，只有当时叶子结点时用于判定
        self.leaf = 0    # 当前结点为叶子结点时指定为1
        self.prior = 0  # 当进行决策时出现缺失值，优先选择的种类


# ### 2. 决策树定义

# In[5]:


class decisionTree:
    def __init__(self):
        self.root = Node('')
    def createTree(self, attributes, datas): # 当前可用属性，当前可用样本，当前所在的递归层数（即在第几层结点）
        node = Node('')
        node.kind = self.getKind(datas)
        sameFlag = 1    # 标记当前样本种类是否相同
        for i in range(1, len(datas)):
            if datas[i]['survived'] != datas[0]['survived']:
                sameFlag = 0
                break
        if sameFlag == 1:        # 递归出口①：当样本属于同一类别
            node.leaf = 1
            return node
        
        delAttributes = []   # 需要删除的无效划分属性
        for a in attributes:
            if a == 'pclass' or a == 'sibsp' or a == 'parch':
                effectiveAttribute = [0, 0, 0]   # 标记当前属性是否为有效属性
                for data in datas:
                    effectiveAttribute[data[a]] = 1   # 说明该属性有样本
                if effectiveAttribute[0] * effectiveAttribute[1] * effectiveAttribute[2] == 0:  # 当该属性的有一个取值无样本，则删除该属性
                    delAttributes.append(a)
            elif a == 'sex' or a == 'cabin':
                effectiveAttribute = [0, 0]   # 标记当前属性是否为有效属性
                for data in datas:
                    effectiveAttribute[data[a]] = 1   # 说明该属性有样本
                if effectiveAttribute[0] * effectiveAttribute[1]== 0:  # 当该属性的有一个取值无样本，则删除该属性
                    delAttributes.append(a)
            elif a == 'embarked':
                effectiveAttribute = [0 , 0, 0, 0]   # 标记当前属性是否为有效属性
                for data in datas:
                    effectiveAttribute[data[a]] = 1   # 说明该属性有样本
                if effectiveAttribute[1] * effectiveAttribute[2] * effectiveAttribute[3] == 0:  # 当该属性的有一个取值无样本，则删除该属性
                    delAttributes.append(a)                           # 不记录缺失值
        for a in delAttributes:   # 从属性列表中删除无效属性
            attributes.remove(a)
        if len(attributes) == 0:   # 递归出口②：如果此时无有效属性
            node.leaf = 1
            return node
        
        gini, a, boundary = self.Gini(attributes, datas)
        node.attribute = a  # 当前结点使用的划分属性
        attributes.remove(a)
        
        if a == 'pclass' or a == 'sibsp' or a == 'parch':
            datasSub = [[],[],[]]  # 保存用于划分的子集
            for data in datas:
                datasSub[data[a]].append(deepcopy(data))  # 子集添加元素
            if len(datasSub[0]) == 0 or len(datasSub[1]) == 0 or len(datasSub[2]) == 0:   # 递归出口③：有一个划分样本集合为空，停止划分
                node.leaf = 1
                return node
            for i in range(3):  # 若集合都不为空，则继续递归划分
                node.son.append(self.createTree(deepcopy(attributes), datasSub[i]))
            return node
        
        elif a == 'sex' or a == 'cabin':
            datasSub = [[],[]]  # 保存用于划分的子集
            for data in datas:
                datasSub[data[a]].append(deepcopy(data))  # 子集添加元素
            if len(datasSub[0]) == 0 or len(datasSub[1]) == 0 :   # 递归出口③：有一个划分样本集合为空，停止划分
                node.leaf = 1
                return node
            for i in range(2):  # 若集合都不为空，则继续递归划分
                node.son.append(self.createTree(deepcopy(attributes), datasSub[i]))
            return node
        
        elif a == 'fare':
            node.boundary = boundary  # 由于是连续值，需要调整
            datasSub = [[],[]]
            for data in datas:
                if data[a] < boundary:
                    datasSub[0].append(deepcopy(data))   # 添加相应的权重
                else:
                    datasSub[1].append(deepcopy(data))   # 添加相应的权重
            if len(datasSub[0]) == 0 or len(datasSub[1]) == 0 :   # 递归出口③：有一个划分样本集合为空，停止划分
                node.leaf = 1
                return node
            for i in range(2):  # 若集合都不为空，则继续递归划分
                node.son.append(self.createTree(deepcopy(attributes), datasSub[i]))
            return node
        
        elif a == 'embarked':
            datasSub = [[],[],[]]  # 保存用于划分的子集
            missData = []  # 保存缺失值
            for data in datas:
                if data[a] != 0:
                    datasSub[data[a] - 1].append(deepcopy(data))  # 子集添加元素
                else:
                    missData.append(deepcopy(data))  # 添加缺失值
            length = []
            length.append(len(datasSub[0]))  # 记录各个集合的大小，用于后续计算
            length.append(len(datasSub[1]))
            length.append(len(datasSub[2]))
            lenSum = sum(length)
            lenMax = max(length)
            if length[0] * length[1] * length[2] == 0:   # 递归出口③：有一个划分样本集合为空，停止划分
                node.leaf = 1
                return node
            if lenMax == length[0]:  # 由于embarked属性有可能出现缺失值，所有要设置优先属性
                node.prior = 0
            elif lenMax == length[1]:
                node.prior = 1
            else:
                node.prior = 2
            for data in missData:  # 将缺失值调整权重加入到各个集合
                for i in range(3):
                    temp = deepcopy(data)
                    temp['w'] *= length[i]/lenSum   # 修改权重
                    datasSub[i].append(temp)
            for i in range(3):  # 若集合都不为空，则继续递归划分
                node.son.append(self.createTree(deepcopy(attributes), datasSub[i]))
            return node
        
        elif a == 'age':
            node.boundary = boundary
            datasSub = [[],[]]
            missData = []  # 保存缺失值
            for data in datas:
                if data[a] != 0:
                    if data[a] < boundary:
                        datasSub[0].append(deepcopy(data))   # 添加相应的权重
                    else:
                        datasSub[1].append(deepcopy(data))   # 添加相应的权重
                else:
                    missData.append(deepcopy(data))  # 添加缺失值
            length = []
            length.append(len(datasSub[0]))  # 记录各个集合的大小，用于后续计算
            length.append(len(datasSub[1]))
            lenSum = sum(length)
            lenMax = max(length)
            if len(datasSub[0]) == 0 or len(datasSub[1]) == 0 :   # 递归出口③：有一个划分样本集合为空，停止划分
                node.leaf = 1
                return node
            if lenMax == length[0]:  # 由于embarked属性有可能出现缺失值，所有要设置优先属性
                node.prior = 0
            else:
                node.prior = 1
            for data in missData:  # 将缺失值调整权重加入到各个集合
                for i in range(2):
                    temp = deepcopy(data)
                    temp['w'] *= length[i]/lenSum   # 修改权重
                    datasSub[i].append(temp)
            
            for i in range(2):  # 若集合都不为空，则继续递归划分
                node.son.append(self.createTree(deepcopy(attributes), datasSub[i]))
            return node
            
        
    def getKind(self, datas):
        count = 0
        for data in datas:
            count += data['survived']
        if count > len(datas)//2:
            return 1
        else:
            return 0
        
    def Gini(self, attributes, datas):
        giniList = []
        for a in attributes:
            if a == 'pclass' or a == 'sibsp' or a == 'parch':  # 这三类相似，离散属性都是三种取值
                count=[[0,0], [0,0], [0,0]]   # 用于保存该属性下存活于死亡的情况
                for data in datas:
                    count[data[a]][data['survived']] += data['w']   # 添加相应的权重
                gini = 0
                for i in range(3):  # 计算基尼指数
                    gini += (count[i][0] + count[i][1])/len(datas) * (1 - (count[i][1]/(count[i][0] + count[i][1]))**2)
                giniList.append(gini)
            
            elif a == 'sex' or a == 'cabin':  # 这两类相似，离散数学都是两种取值
                count=[[0,0], [0,0]]    # 用于保存该属性下存活于死亡的情况
                for data in datas:
                    count[data[a]][data['survived']] += data['w']   # 添加相应的权重
                gini = 0
                for i in range(2):  # 计算基尼指数
                    gini += (count[i][0] + count[i][1])/len(datas) * (1 - (count[i][1]/(count[i][0] + count[i][1]))**2)
                giniList.append(gini)
                
            elif a == 'fare':
                fareList = []
                for data in datas:
                    fareList.append(data['fare'])   # 添加所有的fare
                fareList = list(set(fareList))   # 去重
                fareList.sort()
                for i in range(len(fareList) - 1):
                    fareList[i] = (fareList[i] + fareList[i+1])/2   # 计算所有可能的中位值
                fareList.pop()
                gini_temp = []  # 暂存所有的gini指数
                for fare in fareList:
                    count=[[0,0], [0,0]]
                    for data in datas:
                        if data['fare'] < fare:
                            count[0][data['survived']] += data['w']   # 添加相应的权重
                        else:
                            count[1][data['survived']] += data['w']   # 添加相应的权重
                    gini = 0
                    for i in range(2):  # 计算基尼指数
                        gini += (count[i][0] + count[i][1])/len(datas) * (1 - (count[i][1]/(count[i][0] + count[i][1]))**2)
                    gini_temp.append(gini)
                gini = min(gini_temp)   # 求出最小的基尼指数
                fare = fareList[gini_temp.index(gini)]   # 求出最小基尼指数相应的划分fare
                giniList.append(gini)
            
            elif a == 'embarked':
                count=[[0,0], [0,0], [0,0]]   # 用于保存该属性下存活于死亡的情况
                dataNum = 0
                for data in datas:
                    if data['embarked'] != 0:  # 不是缺失值情况
                        count[data['embarked'] - 1][data['survived']] += data['w']   # 添加相应的权重
                        dataNum += 1  # 非缺失值加1
                rho = dataNum/len(datas)
                gini = 0
                for i in range(3):  # 计算基尼指数
                    gini += (count[i][0] + count[i][1])/len(datas) * (1 - (count[i][1]/(count[i][0] + count[i][1]))**2)
                gini *= rho   # 乘以rho
                giniList.append(gini)
                
            elif a == 'age':
                ageList = []
                for data in datas:
                    if data['age'] != 0:  # 当不是缺失值时
                        ageList.append(data['age'])   # 添加所有的age
                ageNum = len(ageList)
                rho = ageNum/len(datas)
                ageList = list(set(ageList))   # 去重
                ageList.sort()
                for i in range(len(ageList) - 1):
                    ageList[i] = (ageList[i] + ageList[i+1])/2   # 计算所有可能的中位值
                ageList.pop()
                gini_temp = []  # 暂存所有的gini指数
                for age in ageList:
                    count=[[0,0], [0,0]]
                    for data in datas:
                        if data['age'] != 0:
                            if data['age'] < age:
                                count[0][data['survived']] += data['w']   # 添加相应的权重
                            else:
                                count[1][data['survived']] += data['w']   # 添加相应的权重
                    gini = 0
                    for i in range(2):  # 计算基尼指数
                        gini += (count[i][0] + count[i][1])/len(datas) * (1 - (count[i][1]/(count[i][0] + count[i][1]))**2)
                    gini *= rho   # 乘以rho
                    gini_temp.append(gini)
                gini = min(gini_temp)   # 求出最小的基尼指数
                age = ageList[gini_temp.index(gini)]   # 求出最小基尼指数相应的划分age
                giniList.append(gini)
            
            gini = min(giniList)  # 求出所有划分可能中最小的基尼指数
            a = attributes[giniList.index(gini)]  # 求出对应的划分属性
            
            if a == 'age':
                return gini, a, age # 连续值情况下，返回对应的划分边界
            elif a =='fare':
                return gini, a, fare # 连续值情况下，返回对应的划分边界
            else:
                return gini, a, 0
    def predict(self, node, predictData):
        if node.leaf == 1:  # 当前结点为叶子结点时
            return node.kind
        else:
            a = node.attribute
            if a == 'embarked':
                if predictData[a] == 0:    # 当前结点此值为缺失值时
                    return self.predict(node.son[node.prior], predictData)
                else:  # 如果不是缺失值，则按属性划分
                    return self.predict(node.son[predictData[a] - 1], predictData)
            elif a == 'fare':
                if predictData[a] < node.boundary:   # 连续值处理
                    return self.predict(node.son[0], predictData)
                else:
                    return self.predict(node.son[1], predictData)
            elif a == 'age':
                if predictData[a] == 0:        # 当前结点此值为缺失值
                    return self.predict(node.son[node.prior], predictData)
                else:
                    if predictData[a] < node.boundary:   # 连续值处理
                        return self.predict(node.son[0], predictData)
                    else:
                        return self.predict(node.son[1], predictData)
            else:
                return self.predict(node.son[predictData[a]], predictData)
    def postOrderTraverse(self, node, route, traverseList):  # 后根遍历所有的非叶子结点
        for i in range(len(node.son)):
            if node.son[i].leaf != 1:
                temp_route = deepcopy(route)  # 如果其子节点不是叶子结点，则继续递归访问
                temp_route.append(i)
                self.postOrderTraverse(node.son[i], temp_route, traverseList)
        traverseList.append(route)   # 最后添加当前结点路径
    def postPruning(self, curAccuracy, devData):   # 后剪枝
        traverseList = []
        self.postOrderTraverse(node = self.root, route = [],traverseList = traverseList)   # 获取后根遍历的路径
        tempNode = Node('')
        for route in traverseList:  # 依次遍历路径，按照后根遍历的顺序进行后剪枝
            tempNode = self.root
            for i in route:
                tempNode = tempNode.son[i]  # 遍历到目标结点
            tempNode.leaf = 1
            count = 0
            for data in devData:
                if data['survived'] == cartTree.predict(cartTree.root, data):
                    count +=1
            Accuracy = count/len(devData)  # 计算当前在验证集上的正确率
            if Accuracy <= curAccuracy:  # 如果正确率降低了，那么撤回修改
                tempNode.leaf = 0
            else:
                curAccuracy = Accuracy  # 如果正确率上升了，执行修改并且更新当前的准确率
    def showTree(self, node, layer):
        if node.leaf == 0:
            show_str = str(layer)
            for i in range(layer):
                show_str += '-*'
            show_str += node.attribute
            print(show_str)
        for son in node.son:
            self.showTree(son, layer+1)


# ### 3. 决策树初始化

# In[6]:


cartTree = decisionTree()
cartTree.root = cartTree.createTree(attributes = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked'], datas = trainData)
cartTree.showTree(cartTree.root, 1)  # 查看决策树


# ### 4. 在验证集上检验准确率

# In[7]:


count = 0
for data in devData:
    if data['survived'] == cartTree.predict(cartTree.root, data):
        count +=1
Accuracy = count/len(devData)
print('验证集准确率：'+ str(Accuracy))


# ### 5. 进行后剪枝后的准确率

# In[8]:


cartTree.postPruning(curAccuracy = Accuracy, devData = devData)


# In[9]:


count = 0
for data in devData:
    if data['survived'] == cartTree.predict(cartTree.root, data):
        count +=1
Accuracy = count/len(devData)
print('验证集准确率：'+ str(Accuracy))


# ### 6. 查看剪枝后的决策树

# In[10]:


cartTree.showTree(cartTree.root, 1)


# ## 预测测试集

# ### 1. 加载测试集

# In[11]:


def loadTestData(filename):
    data = []
    f = list(csv.reader(open(filename, 'r')))[1:]   # 读取去掉表头的部分
    embarkedDist = {'C':1, 'Q':2, 'S':3}    # 无缺失值时，'C':1, 'Q':2, 'S':3
    sibspParch = [1, 2]
    for line in f:
        if int(line[5]) == 0:  # 转化sibsp
            sibsp = 0
        elif int(line[5]) <= 2:
            sibsp = 1
        else:
            sibsp = 2
        
        if int(line[6]) == 0:   # 转化sibsp
            parch = 0
        elif int(line[6]) <= 2:
            parch = 1
        else:
            parch = 2
        dataDist={'pclass': int(line[1]) - 1,
             'sex': 0 if line[3] == 'male' else 1,   # male:0  female:1
             'age': 0 if len(line[4]) == 0 else float(line[4]),   # 有缺失值保存为0
             'sibsp': sibsp,
             'parch': parch,
             'fare': 0 if len(line[8]) == 0 else float(line[8]),  # 测试集中发现有一个缺失值
             'cabin': 0 if len(line[9]) == 0 else 1,   # 有缺失值：0  无缺失值：1
             'embarked': 0 if len(line[10]) == 0 else embarkedDist[line[10]]}   # 有缺失值：0  无缺失值保存为1、2、3
        data.append(dataDist)
    return data


# ### 2. 进行预测，并将结果写入csv文件

# In[12]:


testData = loadTestData('test.csv')
print(testData[0])


# In[13]:


predict = []
for data in testData:
    predict.append(cartTree.predict(cartTree.root, data))
print(len(predict))


# In[14]:


f = open('testPredict.csv','w',encoding='utf-8',newline='' "")
csv_writer = csv.writer(f)
csv_writer.writerow(['PassengerId','Survived'])
for i in range(418):
    csv_writer.writerow([str(i+892),str(predict[i])])
f.close()


# ### 3.上传Kaggle

# ### 后剪枝处理前
# ![UGPP1I.png](https://s1.ax1x.com/2020/07/12/UGPP1I.png)

# ### 后剪枝处理后
# ![UGU8rq.png](https://s1.ax1x.com/2020/07/13/UGU8rq.png)

# In[ ]:




