#numpy基础



## numpy的优势

python的list支持元素为不同类型，numpy.array只允许元素为同一类型，效率高。

numpy.array把数组当做矩阵，计算灵活方便，python的array不支持矩阵操作。

##numpy.array基础语法

```python
import numpy as np
nparr1 = np.array([1,2,3,4,5,6])
print(nparr1.dtype) 
nparr2 = np.array([1,2,3,4,5,6.0])
print(nparr2.dtype)
nparr = np.array([i for i in range(10)])
print(nparr.size)
运行结果
int32
float64
10

@@@其他创建numpy.array的方法  zeros/ones/full
import numpy as np
nparr1 = np.zeros(shape = 10,dtype = int)#默认就是int
nparr2 = np.full(shape=(3,5),fill_value= 4,dtype = float)
nparr3 = np.ones(shape=(3,5) ,dtype = float)


@@@其他创建numpy.array的方法  arange/linspace
import numpy as np
nparr1 =  np.arange(1,20,3)#步长可为整数，不写步长默认为1，不写起始默认为0
print(nparr1)
nparr2 = np.arange(0,2,0.2)#步长可为浮点数
print(nparr2)
运行结果
[ 1  4  7 10 13 16 19]
[0.  0.2 0.4 0.6 0.8 1.  1.2 1.4 1.6 1.8]

n1 = np.linspace(0,10,5)#在0-10之间均匀取5个数，linear space线性间隔
print(n1) #[ 0.   2.5  5.   7.5 10. ]

其他创建numpy.array的方法  random模块的randint/random方法
import numpy as np
#random模块 randint
n3 = np.random.randint(100)#产生一个0-100的随机整数
print(n3)

n1 = np.random.randint(0,10,size = 10)
n2 = np.random.randint(0,10,size = (4,3))
print(n1)#[8 4 5 9 2 0 5 2 1 9]
print(n2)
'''[[7 0 2]
 [6 6 6]
 [5 3 6]
 [2 9 9]]'''

#random模块 seed 随机种子，设定一个数之后，下次可得同样的随机结果
np.random.seed(456)
np1 = np.random.randint(100)
print(np1)#27

np.random.seed(456)
np2 = np.random.randint(100)
print(np2)

#random模块 random()方法
np3 = np.random.random()#产生1个随机小数，省去的参数是size
print(np3)#0.980484151565036
np4 = np.random.random(5)#产生5个随机小数
print(np4)#[0.15154137 0.33468878 0.28516385 0.15465477 0.31205261]
np5 = np.random.random((3,3))
print(np5)
'''[[0.12135984 0.81765609 0.16459023]
 [0.42800557 0.33141084 0.67745542]
 [0.71277191 0.09191151 0.03126743]]'''

#random模块 normal()方法，生成正态分布的数据
np1 = np.random.normal(loc=1,scale=2,size=4)
参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
参数size(int 或者整数元组)：输出的值赋在shape里，默认为None。
```



##numpy.array的基本操作

```python
import numpy as np
x = np.arange(10)
print(x)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X = np.arange(15).reshape((3,5))
print(X)

#numpy.array的基本属性 ndim/size/shape
print(x.ndim)
print(X.shape)
#numpy.array的访问 --切片访问,不改变原数组
print(x[3:6])
print(x[::2])
print(X[0, 3])
print(X[0:2, 1:3])
x[:,0]第一列的元素
x[0,:]第一行的元素

切片访问相当于 ==建立了索引== ，切片不改变原数组，改变切片会同样改变原数组
copy（）方法是复制了一份，切片与原数组毫无关系，改变不会互相影响，如下
subX = X[:2,:3].copy()#这是subX与X无关系

#numpy.array的变形 -- reshape 不改变原数组
x.reshape(2,5)
x.reshape(-1,5)# 写成-1 则会自动根据另一个数计算
```

## numpy.array合并与分割

### concatenate/vstack/hstack

NumPy中的数组轴是编号的，从0开始类似于Python中元组，列表，字符串的索引

**Numpy轴是我们执行操作的方向**

np.sum（），np.mean（），np.min（），np.median（）与axis一起使用 指定的是折叠的轴

np.concatenate()和axis一起使用指定的是 堆叠的轴，即拼接的轴

```python
import numpy as np
x = np.arange(9).reshape(3,-1)
X = np.arange(15).reshape(3,5)
r1 = np.concatenate((x,X),axis=1)
r2 = np.hstack(x,X)
#vstack 在垂直轴进行操作，hstack在水平轴进行操作
```

### split/vsplit/hsplit

注意：切片访问和数组分割的区别，数组分割是不能得到一个二维矩阵的左上角，即一个二维矩阵的1/4的

```python
#一维数组时：第一个参数指定分割谁，第二个参数指定分割点
x = np.arange(15)
x1,x2,x3 = np.split(x,[3,5])
#二维数组时：第一个参数指定分割谁，第二个参数列表指定默认维度（0）的分割点
A = np.arange(16).reshape((4, 4))
a1 ,a2 = np.split(A,[2])
#二维数组时：同时指定axis，第一个参数指定分割谁，第二个参数列表是axis指定的维度的分割点
A = np.arange(16).reshape((4, 4))
a1 ,a2,a3，a4 = np.split(A,[1，2,3],axis=0)
```

##numpy.array 的一般计算

（1）加减乘除，逻辑非，**表示的指数运算符和%表示的模运算符  都是一元通用函数

（2）绝对值函数。直接abs()，括号内为一个Numpy数组

（3）三角函数。np.sin() np.cos() np.tan() np.arcsin()  np.arccos() np.arctan() 括号里直接是一个Numpy数组

（4）指数和对数。np.exp(x)表示 e^x ，  np.exp2(x)表示2^x  np.power(3,x)表示3^x。

​          最基本的 np.log 给出的是以自然数为底数的对数。如果你希望计算以2为底数或者以10为底数的对数 np.log2()，np.log10()

```python
import numpy as np
X = np.arange(1,16).reshape(3,5)
#X +1 ;X-1;X*2;X/2;X//2取整;X%2取余；X**2 幂；1/X倒数
np.abs(X)；
np.sin(X)；
np.arctan(X)；
np.exp(X)；#np.exp()函数是求  e的x次方。
np.exp2(X)#2的x次方
np.power(3, X)；#3的X次方
np.log(X)
np.log2(X)
np.log10(X)
```

## numpy.array矩阵运算

两个矩阵之间的加减乘 ，点乘和转置

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201118171643044.png" alt="image-20201118171643044" style="zoom:80%;" />



## 向量和矩阵的运算，矩阵的逆



<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201119082620621.png" alt="image-20201119082620621" style="zoom:80%;" />

## numpy中的聚合操作：min/max/sum

注意：axis描述的是将要被压缩的维度

```python
#一维聚合
big_array = np.random.rand(1000000)
np.sum(big_array)
np.min(big_array)
np.max(big_array)
#多维聚合，加参数 axis
X = np.arange(16).reshape(4,-1)#4行4列
np.sum(X,axis = 0)#在0维压缩

#其他聚合操作
①np.prod()返回给定维度上各个元素的乘积
numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>)
axis是指求积的维度
keepdims是指保持维度，不缩减
initial是起始数，即返回的矩阵会在元素乘积上再乘起始数
>>> np.prod([[1.,2.],[3.,4.]])
24.0
>>> np.prod([[1.,2.],[3.,4.]], axis=1)
array([  2.,  12.])
>>> x = np.array([1, 2, 3], dtype=np.uint8)
>>> np.prod(x).dtype == np.uint
True
>>> np.prod([1, 2], initial=5)
10

②/mean/median/std/var/percentile
X = np.arange(16).reshape(4,-1)
print(X)
print(np.mean(X))#平均值
print(np.median(X))#中位数
print(np.std(X))#标准差
print(np.var(X))#方差
#计算百分位,当q=50，相当于median, q=100相当于求最大值
print(np.percentile(X, q=50))
```

##numpy中的arg运算

“argument of the maximum/minimum

argmax f(x): 当f(x)取最大值时，x的取值”

```python
x = np.random.normal(0,1,1000)#均值为0，方差为1的正态分布
np.argmax(x)#返回最大值的索引值
np.argmin(x)#返回最小值的索引值

#排序操作
np.random.shuffle(x)#打乱顺序，改变原数组
np.sort(x)#排序，不改变原数组
x.sort()#python内置函数，排序，改变原数组
np.argsort(x)#返回排序后的索引序列，不改变原数组

#partition操作
np.partition(x,3)#左边都比3小，右边都比3大
np.agepartition(x,3)#索引
```

##numpy中的比较和Fancy indexing

```python
import numpy as np
#fancy indexing 在一维数组中
x = np.arange(16)
x[3]#3
x[3:9]
x[3:9:2]

ind = [3,5,7]
print(x[ind])#[3 5 7]

ind = np.array([[3,2],[1,3]])
print(x[ind])
'''[[3 2]
 [1 3]]''
```

fancy indexing 在二维数组中 

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201119111107726.png" alt="image-20201119111107726" style="zoom:80%;" />

### `numpy.array` 的比较

In [17]:

```
x
```

Out[17]:

```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
```

In [18]:

```
x < 3
```

Out[18]:

```
array([ True,  True,  True, False, False, False, False, False, False,
       False, False, False, False, False, False, False], dtype=bool)
```

In [19]:

```
x > 3
```

Out[19]:

```
array([False, False, False, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True], dtype=bool)
```

In [20]:

```
x <= 3
```

Out[20]:

```
array([ True,  True,  True,  True, False, False, False, False, False,
       False, False, False, False, False, False, False], dtype=bool)
```

In [21]:

```
x >= 3
```

Out[21]:

```
array([False, False, False,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True], dtype=bool)
```

In [22]:

```
x == 3
```

Out[22]:

```
array([False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False], dtype=bool)
```

In [23]:

```
x != 3
```

Out[23]:

```
array([ True,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True,  True], dtype=bool)
```

In [28]:

```
2 * x == 24 - 4 * x
```

Out[28]:

```
array([False, False, False, False,  True, False, False, False, False,
       False, False, False, False, False, False, False], dtype=bool)
```

In [29]:

```
X < 6
```

Out[29]:

```
array([[ True,  True,  True,  True],
       [ True,  True, False, False],
       [False, False, False, False],
       [False, False, False, False]], dtype=bool)
```

### 使用 `numpy.array` 的比较结果

In [30]:

```
np.count_nonzero( x <= 3)
```

Out[30]:

```
4
```

In [32]:

```
np.sum(x <= 3)
```

Out[32]:

```
4
```

In [33]:

```
np.sum(X % 2 == 0, axis=0)
```

Out[33]:

```
array([4, 0, 4, 0])
```

In [34]:

```
np.sum(X % 2 == 0, axis=1)
```

Out[34]:

```
array([2, 2, 2, 2])
```

In [37]:

```
np.any(x == 0)
```

Out[37]:

```
True
```

In [38]:

```
np.any(x < 0)
```

Out[38]:

```
False
```

In [39]:

```
np.all(x > 0)
```

Out[39]:

```
False
```

In [40]:

```
np.all(x >= 0)
```

Out[40]:

```
True
```

In [41]:

```
np.all(X > 0, axis=1)
```

Out[41]:

```
array([False,  True,  True,  True], dtype=bool)
```

In [44]:

```
np.sum((x > 3) & (x < 10))
```

Out[44]:

```
6
```

In [45]:

```
np.sum((x > 3) && (x < 10))
  File "<ipython-input-45-780ca9b7c144>", line 1
    np.sum((x > 3) && (x < 10))
                    ^
SyntaxError: invalid syntax
```

In [46]:

```
np.sum((x % 2 == 0) | (x > 10))
```

Out[46]:

```
11
```

In [47]:

```
np.sum(~(x == 0))
```

Out[47]:

```
15
```

### 比较结果和Fancy Indexing

In [17]:

```
x < 5
```

Out[17]:

```
array([ True,  True,  True,  True,  True, False, False, False, False,
       False, False, False, False, False, False, False], dtype=bool)
```

In [18]:

```
x[x < 5]
```

Out[18]:

```
array([0, 1, 2, 3, 4])
```

In [20]:

```
x[x % 2 == 0]
```

Out[20]:

```
array([ 0,  2,  4,  6,  8, 10, 12, 14])
```

In [21]:

```
X[X[:,3] % 3 == 0, :]
```

Out[21]:

```
array([[ 0,  1,  2,  3],
       [12, 13, 14, 15]])
```

##matplotlib基础

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
x = np.linspace(0,10,100)
siny = np.sin(x)
cosy = np.cos(x)

plt.plot(x,siny,color = "red" ,label = "siny",linestyle="--")
plt.plot(x,cosy,color = "blue",label = "cosy")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.title("welcome")
plt.show()

plot.scatter(x,siny，marker="o")#绘制散点图，marker = o/+/x

关于color参数：https://matplotlib.org/2.0.2/api/colors_api.html
    方式一：写以下的全称或只写一个首字母
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
方式二：C0~C7
C0: The first color in the cycle
C1: The second color in the cycle
方式三：灰度模式可用小数
color = '0.75'
方式四：
color = '#eeefff'
(r, g, b) or (r, g, b, a) rgba范围都在[0-1]
    
    
关于linestyle参数：https://matplotlib.org/devdocs/gallery/lines_bars_and_markers/line_styles_reference.html
```

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201119142442657.png" alt="image-20201119142442657" style="zoom:80%;" />

## 读取数据集和可视化表示

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201119144349720.png" alt="image-20201119144349720" style="zoom:80%;" />

# KNN模型

## 实现简单的knn

mykNN.py

```python
import numpy as np
from math import sqrt
from collections import Counter
#以下是自己实现简单的knn
def kNN_classify(k,X_train,y_train,x):
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    distances = [ sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)[:k]

    topK_y = y_train[nearest]
    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]



####测试用例
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)
x = np.array([8.093607318, 3.365731514])#预测样本
k = 6

print(kNN_classify(k, X_train, y_train, x))
```

```python
#模仿scikit-learn封装自己的knn
import numpy as np
from collections import Counter
from math import sqrt
class kNNClassifier:
    def __init__(self,n_neighbers):
        assert n_neighbers>1 ,"n_neighbers must be valid"
        self.k = n_neighbers
        self._X_train = None
        self._y_train = None

    def fit(self,_X_train,_y_train):
        """根据训练数据集_X_train和_y_train训练kNN分类器"""
        assert _X_train.shape[0] == _y_train.shape[0], \
            "the size of _X_train must be equal to the size of _y_train"
        assert self.k <= _X_train.shape[0], \
            "the size of _X_train must be at least k."

        self._X_train = _X_train
        self._y_train = _y_train
        return self

    def __predict(self,x_predict):
        '''给定单个待预测数据'''
        assert x_predict.shape[0] == self.__X_train.shape[1], \
            "the feature number of x must be equal to _X_train"

        distances = [sqrt(np.sum((_X_train - x_predict) ** 2)) for _X_train in self._X_train]
        nearest = np.argsort(distances)[:self.k]

        topK_y = self._y_train[nearest]
        votes = Counter(topK_y)
        return votes.most_common()[0][0]

    def predict(self,X_predict):
        '''给定一个待预测数据集，返回预测的结果向量'''
        y_predict = [ self.__predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k
```

## 使用scikit-learn中的kNN

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201119165317050.png" alt="image-20201119165317050" style="zoom:80%;" />



```python
from sklearn.neighbors import KNeighborsClassifier
kNN_classifer=KNeighborsClassifier(n_neighbors=6)
kNN_classifer.fit(X_train,y_train)#用的都是上边的数据X_train,y_train
predict_x = np.reshape(x,(1,-1))
print(kNN_classifer.predict(predict_x)[0])
```

## 训练测试分离—train_test_split

python文件命名中不要有短横线-，会无法import

```python
#可以设置随机种子 random_state
X_train ,X_test,y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
```

```python
#首先学习一个函数：np.random.permutation() 随机排列序列
np.random.permutation(5)#array([4, 1, 0, 3, 2])
np.random.permutation([0,1,2,3,4,5])#array([1, 4, 3, 5, 2, 0])
X = np.arange(9).reshape(3,-1)
np.random.permutation(X)
#array([[3, 4, 5],
#       [6, 7, 8],
#       [0, 1, 2]])
np.random.permutation(X)
#array([[0, 1, 2],
#       [3, 4, 5],
#       [6, 7, 8]])      注意：不改变原数组
================================================================
#用自己的方式封装train-test_split
def train_test_split(X,y,test_ratio):
    import numpy as np
    test_size = int(test_ratio * len(X))  # 注意这里要取整数
    shuffle_index = np.random.permutation(len(X))
    X_train = X[shuffle_index[:test_size]]
    X_test = X[shuffle_index[test_size:]]
    y_train = y[shuffle_index[:test_size]]
    y_test = y[shuffle_index[test_size:]]
    return X_train,X_test,y_train,y_test
#测试自己封装的train-test_split
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,0.2)
kcf = kNNClassifier(6)
kcf.fit(X_train,y_train)
y_predict = kcf.predict(X_test)
print(y_predict)
accuracy =np.sum(y_predict == y_test) / len(y_test)
print(accuracy)
================================================================
#scikitlearn中的train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
```

## 分类准确度

metrics度量

```python
import numpy as np
def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)
```

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201120093557677.png" alt="image-20201120093557677" style="zoom:80%;" />

 

## knn中的超参数

一：定义

　　超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。

二：常用超参数

　　k近邻算法的k，权重weight，明可夫斯基距离公式的p，这三个参数都在KNeighborsClassifier类的构造函数中。

三：共同代码

 

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
 
digits = datasets.load_digits()
 
x = digits.data
y = digits.target
 
#在 train_test_split直接写0.2会对应到别的参数，应该写上test_size
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
```

 

 

四：k的最优数值

```python
best_score = 0.0
best_k = -1
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    t = knn.score(x_test,y_test)
    if t>best_score:
        best_score = t
        best_k = k
 
print(best_k)
print(best_score)
```

 

五：weight的最优数值

　　如果取值为uniform，例如：当我们取k等于3，结果预测到三个点距离最近的点为三个，sklearn就会选择一个进行返回预测结果，但是我们如果考虑距离也就是取值为distance，就会有一个权重的概念，一般为距离的倒数，例如该点到另外三个点的距离为1，3，4则权重为1，1/3，1/4，则返回1这个点作为预测结果。

 

```python
best_score = 0.0
best_k = -1
best_method = ''
for method in ['uniform','distance']:
    for k in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=k,weights=method)
        knn.fit(x_train,y_train)
        t = knn.score(x_test,y_test)
        if t>best_score:
            best_score = t
            best_k = k
            best_method = method
 
print(best_score)
print(best_k)
print(best_method)
```

六：p的最优数值

　　当需要p的参数时，weight必须为distance，不能为uniform

```python
best_score = 0.0
best_k = -1
best_p = 1
for i in range(1,6):
    for k in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=k,weights='distance',p=i)
        knn.fit(x_train,y_train)
        t = knn.score(x_test,y_test)
        if t>best_score:
            best_k = k
            best_score = t
            best_p = i
print(best_p)
print(best_score)
print(best_k)
```

 ## 网格搜索

grid search是用来寻找模型的最佳参数

**用json设置参数**

```python
X_train ,X_test,y_train ,y_test = train_test_split(X,y,test_size=0.2)
#网格搜索
from sklearn.model_selection import GridSearchCV
param_grid = [
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
]
#创建一个空的KNeighborsClassifier
knn_clf = KNeighborsClassifier()
#得到一个网格搜索对象
#在GridSearchCV方法中加上这两个参数可以得到更多详细信息n_jobs=-1, verbose=2
grid_search = GridSearchCV(knn_clf,param_grid)
#进行fit
grid_search.fit(X_train,y_train)
```

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201120101532656.png" alt="image-20201120101532656" style="zoom:80%;" />

**查看最佳分数和最佳参数**　

 ```python
grid.best_score_    #查看最佳分数,grid.best_score_ 是测试集的最佳分数？	                               #grid_search.best_estimator_.score是训练集的最佳分数？
grid.best_params_   #查看最佳参数
 ```

**获取最佳模型**

```python
grid.best_estimator_
```

**利用最佳模型来进行预测**

```python
best_model=grid.best_estimator_
predict_y=best_model.predict(Test_X)
metrics.f1_score(y, predict_y)
```

## 数据归一化处理

**是针对每一个特征维度来做的，而不是针对样本。** 

最值归一化Normalization：这种归一化方法比较适用在数值比较集中的情况。这种方法有个缺陷，如果max和min不稳定，很容易使得归一化结果不稳定，使得后续使用效果也不稳定。实际使用中可以用经验常量值来替代max和min。

![img](numpy%E5%9F%BA%E7%A1%80.assets/20190525210603677.png)

```python
X[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
X[:,1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1]))
```

均值方差归一化Standardization

![img](numpy%E5%9F%BA%E7%A1%80.assets/20190525174040159.png)

```python
X2[:,0] = (X2[:,0] - np.mean(X2[:,0])) / np.std(X2[:,0])
X2[:,1] = (X2[:,1] - np.mean(X2[:,1])) / np.std(X2[:,1])
```

## **数据预处理—scikit-learn中的StandardScaler**

通常指特征工程中的**特征缩放**过程，使用特征缩放，第一，使不同量纲的特征处于同一数量级，减少方差大的影响，使模型更准确，第二，加快收敛速度。

缩放过程可以分为以下几种：

缩放到均值为0，方差为1（**Standardization——**StandardScaler()）

缩放到0和1之间（**Standardization——**MinMaxScaler()）

缩放到-1和1之间（**Standardization——**MaxAbsScaler()）

缩放到0和1之间，保留原始数据的分布（**Normalization——**Normalizer()）

1就是常说的z-score归一化，2是min-max归一化。



使用sklearn中的StandardScaler：

```
from sklearn.preprocessing import StandardScaler
```

1.创建StandardScaler()对象并保存

```
standardScalar = StandardScaler()
```

2.fit

3.用StandardScaler对象的transform方法传入需要归一化的数据集，并把结果进行保存

```
X_train = standardScalar.transform(X_train)
```

**归一化的理由：**

- 归一化后加快了梯度下降求最优解的速度；

如果机器学习模型使用梯度下降法求最优解时，归一化往往非常有必要，否则很难收敛甚至不能收敛。

- 归一化有可能提高精度；

一些分类器需要计算样本之间的距离（如欧氏距离），例如KNN。如果一个特征值域范围非常大，那么距离计算就主要取决于这个特征，从而与实际情况相悖（比如这时实际情况是值域范围小的特征更重要）。

**哪些机器学习算法不需要(需要)做归一化?**

 概率模型（树形模型）不需要归一化，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率，如决策树、RF。而像Adaboost、SVM、LR、Knn、KMeans之类的最优化问题就需要归一化。

StandardScaler对象的几个常用方法：

| [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit)(X[, y]) | 计算用于以后缩放的mean和std |
| ------------------------------------------------------------ | --------------------------- |
| [`fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit_transform)(X[, y]) | 适合数据，然后转换它        |
| [`transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.transform)(X[, y, copy]) | 通过居中和缩放执行标准化    |

```python
import numpy as np
#实现自己封装的StandardScaler
class StandardScaler:
    def __init__(self):
        self.mean_ = None#均值
        self.scale_= None#标准差 ，文档说scale是np.sqrt(var_)算的
    def fit(self,X):
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

    def transform(self,X):       
        res = np.empty(X.shape)
        for i in range(X.shape[1]):
            res[:,i] = (X[:,i] - self.mean_[i]) / self.scale_[i]
        return res
```



# 线性回归

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123084159691.png" alt="image-20201123084159691" style="zoom:50%;" />

##简单线性回归的实现

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123084306021.png" alt="image-20201123084306021" style="zoom: 50%;" />

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123083320536.png" alt="image-20201123083320536" style="zoom: 50%;" />

 numerator    分子 ；denominator 分母

补充：python  中 zip（）函数的使用方法：

```python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```

```python
#实现简单的线性回归算法
#对下面这些点用线性回归法求模型
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])
x_mean = np.mean(x)
y_mean = np.mean(y)
numerator = 0.0
denominator = 0.0
for x_i ,y_i in zip(x,y):
    numerator += (x_i - x_mean)*(y_i - y_mean)
    denominator += (x_i - x_mean)**2
a = numerator / denominator
b = y_mean - a * x_mean
y_hat = a*x + b
#画图
plt.scatter(x,y)
plt.plot(x,y_hat,color = 'r')
plt.show()
```



## 向量化运算

用for循环的效率比较低，如果能转成向量之间的运算，效率会提高。

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123083809389.png" alt="image-20201123083809389" style="zoom:50%;" />





<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123083840923.png" alt="image-20201123083840923" style="zoom:50%;" />

```python
#将上一个代码的 分子分母 换成用向量计算的即可
numerator = (x_train - x_mean).dot(y_train - y_mean)
denominator = (x_train - x_mean).dot(x_train- x_mean)
```

```python
#向量化实现的性能测试
m = 1000000
big_x = np.random.random(size=m)
#np.random.normal(size=m)干扰项
big_y = big_x * 2 + 3 + np.random.normal(size=m)

%timeit reg1.fit(big_x, big_y)#for 循环
%timeit reg2.fit(big_x, big_y)#向量化
```



## 回归算法的性能评价MSE/RMSE/MAE

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123085020221.png" alt="image-20201123085020221" style="zoom:50%;" />



将m除掉，让回归算法的性能和样本数量无关，比如：一个算法10个样本，误差是800，另一个算法1000个样本，误差是1000。所以得到以下式子：

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123085145822.png" alt="image-20201123085145822" style="zoom:50%;" />

为了不改变量纲，通常选择将MSE开根号，使其量纲与y保持一致。比如：y的单位是（万元），y的平方就是（万元）的平方，MSE的单位也是（万元） 的平方，这样不符合人们的使用习惯。

**使用MSE还是RMSE，关键看最后的要的结果对量纲是否敏感。**

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123085507433.png" alt="image-20201123085507433" style="zoom:50%;" />



还可以使用MAE。在训练模型的时候不使用绝对值，因为绝对值不是连续可导的函数，但是在评价时可以使用绝对值来评价。

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123090223510.png" alt="image-20201123090223510" style="zoom:50%;" />

可能由于某种原因设置了上限，所以这些点可能不够真实，计算时选择把他们去掉。

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123090759715.png" alt="image-20201123090759715" style="zoom:50%;" />

```python
#封装线性回归的三个误差测评函数 MSE/RMSE/MAE,按公式写即可
def mean_squared_error(y_test,y_predict):
    return  np.sum((y_test - y_predict)**2) /len(y_test)
def root_mean_squared_error(y_test,y_predict):
    return sqrt(np.sum((y_test - y_predict)**2) /len(y_test))
def mean_absolute_error(y_test,y_predict):
    return np.sum(np.abs(y_test - y_predict)) /len(y_test)
```

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123153300660.png" alt="image-20201123153300660" style="zoom:80%;" />

## 最好的性能评价指标 R Squared

因为RMSE/MAE都不是一个具体的分数，不能在多种应用之间进行比较。所以sklearn中用了不同的方法计算并实现score（）。

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123093655849.png" alt="image-20201123093655849" style="zoom:50%;" />

R<sup>2</sup>的意义：<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123094120890.png" alt="image-20201123094120890" style="zoom:50%;" />

R<sup>2</sup>的一些归律：<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123094308712.png" alt="image-20201123094308712" style="zoom:50%;" />

R<sup>2</sup>的变形：

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123094530400.png" alt="image-20201123094530400" style="zoom:50%;" />

```python
#封装R squard
def r2_score(y_test,y_predict):
   return 1- (mean_squared_error(y_test,y_predict) / np.var(y_test))
```



## 多元线性回归和正规方程解

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123095403424.png" alt="image-20201123095403424" style="zoom:50%;" />

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123095509679.png" alt="image-20201123095509679" style="zoom:50%;" />

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123100139546.png" alt="image-20201123100139546" style="zoom:50%;" />![image-20201123101355841](numpy%E5%9F%BA%E7%A1%80.assets/image-20201123101355841.png)

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123101421261.png" alt="image-20201123101421261" style="zoom:50%;" />

```python
#仿照sklearn 实现简单线性回归模型,可适用于多元特征
import numpy as np
from metric import  r2_score
def __init__(self):
    """初始化Linear Regression模型"""
    self.coef_ = None
    self.intercept_ = None
    self._theta = None


def fit_normal(self, X_train, y_train):
    """根据训练数据集X_train, y_train训练Linear Regression模型"""
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must be equal to the size of y_train"

    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

    self.intercept_ = self._theta[0]
    self.coef_ = self._theta[1:]

    return self


def predict(self, X_predict):
    """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
    assert self.intercept_ is not None and self.coef_ is not None, \
        "must fit before predict!"
    assert X_predict.shape[1] == len(self.coef_), \
        "the feature number of X_predict must be equal to X_train"

    X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    return X_b.dot(self._theta)


def score(self, X_test, y_test):
    """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

    y_predict = self.predict(X_test)
    return r2_score(y_test, y_predict)


def __repr__(self):
    return "LinearRegression()"
```

# 梯度下降法

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123214324212.png" alt="image-20201123214324212" style="zoom:50%;" />

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201124143641164.png" alt="image-20201124143641164" style="zoom:50%;" />



```python
#模拟梯度下降法
plot_x = np.linspace(-1,6,141)
plot_y = (plot_x -2.5)**2-1
#绘制一下看看
# plt.plot(plot_x,plot_y)
# plt.show()

#对损失函数J求导的方法
def dJ(theta):
    return 2*(theta - 2.5)
def J(theta):
    return (theta -2.5)**2-1

eta = 0.1
theta = 0
epsilon = 1e-8

while True :
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient

    if(np.abs(J(theta) - J(last_theta))< epsilon):
        break
print(theta)
print(J(theta))
```



## 多元线性回归中的梯度下降法

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201123220637241.png" alt="image-20201123220637241" style="zoom:50%;" />

<img src="numpy%E5%9F%BA%E7%A1%80.assets/image-20201124163731740.png" alt="image-20201124163731740" style="zoom:50%;" />