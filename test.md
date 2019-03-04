```python
1.1	bulabula
1.2	Numpy的基本操作
1.2.1	Numpy介绍
NumPy（Numerical Python）是 Python 中的一个线性代数库，它为 Python 提供高性能向量、矩阵和高维数据结构的科学计算。它通过 C 和 Fortran 实现，因此在用向量和矩阵建立方程并实现数值计算时有非常好的性能。它对每一个数据科学或机器学习 Python 包而言，这都是一个非常重要的库，SciPy（Scientific Python）、Mat-plotlib（plotting library）、Scikit-learn 等都在一定程度上依赖 NumPy。PaddlePaddle在Pip安装时会自动安装对应版本的Numpy依赖。
对数组执行数学运算和逻辑运算时，NumPy 是非常有用的。在用 Python 对 n 维数组和矩阵进行运算时，NumPy 提供了大量有用特征。在使用 PaddlePaddle时，Numpy 不仅仅是一个库，它还是实现深度学习数据表示的基础之一。因此了解它的工作原理、关注向量化和广播（broadcasting）是非常必要的。
这一章节介绍了数据科学初学者需要了解的 NumPy 基础知识，包括如何创建 NumPy 数组、如何使用 NumPy 中的广播机制、如何获取值以及如何操作数组。更重要的是，大家可以通过本文了解到 NumPy 在 Python 列表中的优势：更简洁、更快速地读写项、更方便、更高效。

1.3安装 NumPy
你在安装PaddlePaddle时，PaddlePaddle的安装程序会自动在当下环境集成适合该PaddlePaddle版本的Numpy包。PaddlePaddle的安装方法在第三章中介绍。如果你想单独安装Numpy，那么你可以使用以下命令从终端上安装 NumPy：
pip install numpy
如果你已经装有 Anaconda，那么你可以使用以下命令通过终端或命令提示符安装 NumPy：
conda install numpy

1.4 NumPy 数组
NumPy 数组是包含相同类型值的网格。NumPy 数组有两种形式：向量和矩阵。在计算机科学中，向量是一维数组，矩阵是多维数组。在某些情况下，矩阵也可以只有一行或一列。
在使用Numpy之前先赋予包别名：
import numpy as np
使用Python 列表的创建 NumPy 数组
我们先创建一个 Python 列表“my_list”：
first_list = [1, 2, 3, 4, 5]
通过这个列表，我们可以简单地创建一个名为 my_numpy_list 的 NumPy 数组，显示结果：
one_dimensional_list = np.array(first_list)
one_dimensional_list  #这里将回显由刚刚first_list数组生成的结果
刚才将一个 Python 列表转换成了一维NumPy数组。要想得到二维数组，我们要创建一个列表为元素的列表，如下所示：
second_list = [[1,2,3], [5,4,1], [3,6,7]]
two_dimensional_list = np.array(second_list)
two_dimensional_list  #这里将回显由刚刚second_list数组生成的结果
array ([[1, 2, 3],
         [5, 4, 1],
         [3, 6, 7]])


使用 arange() 内置函数创建 NumPy 数组
NumPy可以用 arange() 创建一个数组，这点与 Python 的 range() 内置函数相似：
first_list = np.arange(10)
或者：
first_list = np.arange(0,10)
这就产生了 0~9 的十个数字：
first_list
array ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


要注意的是 np.arange() 函数中有三个参数。第一个参数表示起始位置，第二个参数表示终止位置，第三个参数表示步长。例如，要得到 0~10 中的偶数，只需要将步长设置为 2 就可以了，如下所示：
first_list = np.arange(0,11,2)
first_list
array ([ 0,  2,  4,  6,  8, 10])
还可以创建有 7 个 0 的一维数组：
my_zeros = np.zeros(7)
my_zeros
array ([0., 0., 0., 0., 0., 0., 0.])
也可以创建有 5 个 1 的一维数组：
my_ones = np.ones(5)
my_ones
array ([1., 1., 1., 1., 1.])
同样，可以生成内容都为 0 的7行 5 列二维数组：
Two-dimensional_zeros = np.zeros((7,5))
Two-dimensional_zeros
array ([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])

使用 linspace() 内置函数创建 NumPy 数组
linspace() 函数返回在指定范围内，具有指定的间隔的数字。也就是说，如果想要 0 到 12 中间隔相等的 4 个点，可以使用以下命令：
isometry_arr = np.linspace(0, 12, 4)
isometry_arr
该命令将结果生成一维向量：
array ([ 0.,  4.,  8., 12.])
与 arange() 函数不同，linspace() 的第三个参数是要创建的数据点数量。


在 NumPy 中创建一个单位矩阵
单位矩阵也叫恒等矩阵、纯量矩阵。处理线性变换时，单位矩阵是非常有用的，它表示无缩放、旋转或平移的变换。一般而言，单位矩阵是一个二维方矩阵，也就是说在这个矩阵中列数与行数相等。单位矩阵的特点是它的对角线数都是 1，其他的都是 0。创建单位矩阵一般只有一个参数，下述命令说明了要如何创建单位矩阵：
identity_matrix = np.eye(6)    #6是矩阵的变长

用 NumPy 创建一个由随机数组成的数组
一般random系函数比较常用的有 rand()、randn() 或 randint() random_integers()，它们都具备生成随机数的功能。
使用 random.rand()，可以以给定的形状创建一个数组，并在数组中加入在[0,1]之间均匀分布的随机样本。
如果想要由 4 个对象组成的一维数组，并使得这 4 个对象均匀分布在 0~1，可以这样做：
my_rand = np.random.rand(4)
my_rand
array([0.54530499, 0.4791477 , 0.17816267, 0.98061916])
如果想要一个有 3 行 4 列的二维数组，则：
my_rand = np.random.rand(3, 4)
my_rand
array([[3.64058527e-01, 9.05102725e-01, 3.25318028e-01, 4.86173815e-01],
       [6.85567784e-01, 7.30340885e-02, 1.36020526e-01, 3.13770036e-04],
       [2.76068271e-02, 5.37804406e-01, 6.09760670e-01, 9.03652017e-01]])

使用 randn()，可以创建一个期望为0，方差为1的标准正态分布（高斯分布）的随机样本。例如，从中生成 30 个服从标准正态分布的随机数：

my_randn = np.random.randn(30)
my_randn
 
array([ 0.46344253, -1.1180354 , -0.76683867,  0.60764125,  0.75040916,
        0.52247857,  1.05988275, -0.40201072, -0.21179046, -0.17263014,
        1.3185744 ,  0.59589626,  1.24200835, -0.80713838,  2.07958112,
        1.37557692,  1.35925843, -0.05960489,  1.26046288,  0.88368104,
        0.30442813,  2.57724599, -0.94821606,  0.37336274, -1.1968936 ,
        1.10085423,  0.3339304 ,  0.63401235,  0.6585172 ,  0.72375082])
如需将其表示为 3 行 5 列的二维数组，这样做即可：
np.random.randn(3,5)
使用 randint() 函数生成整数数组。randint() 函数最多可以有三个参数：最小值（包含、默认为0），最大值（不包含、必填）以及数组的大小（默认为1）。
np.random.randint(5, 20, 7) 
array([10, 12, 19, 12,  8, 13, 14])

将一维数组转换成二维数组
创建一个有 20 个随机整数的一维数组：
arr = np.random.rand(20)
然后使用 reshape() 函数将其转换为二维数组：
arr = arr.reshape(4,5)
array([[0.85161986, 0.06722657, 0.22270304, 0.60935757, 0.20345998],
       [0.67193271, 0.27533643, 0.30484289, 0.78642633, 0.7400095 ],
       [0.63484647, 0.48679984, 0.93656238, 0.81573558, 0.22958044],
       [0.57825764, 0.79502777, 0.77810231, 0.37802153, 0.6360811 ]])
注意：reshape() 转换时，需保持行列数相乘后要与元素数量相等。
假设存在大量数组，而你需要弄清楚数组的形状，只需要使用 shape 函数即可：
arr.shape
(4, 5)
定位 NumPy 数组中的最大值和最小值
使用 max() 和 min() 函数，我们可以得到数组中的最大值或最小值：
arr_2 = np.random.randint(0, 20, 10) #在0到20中随机10个数字 
array([ 8,  9, 13, 13,  1, 14,  8,  0, 17, 18])
arr_2.max() #返回最大的数字为18
arr_2.min() #返回最小的数字为0
使用 argmax() 和 argmin() 函数，我们可以定位数组中最大值和最小值的下标：
arr_2.argmax() #返回最大的数字下标为9
arr_2.argmin() #返回最小的数字下标为7
从 NumPy 数组中索引／选择多个元素（组）
在 NumPy 数组中进行索引与 Python 类似，只需在方括号指定下标即可：
my_array = np.arange(0,13)
my_array[8]  
8
想要获得数组中的一系列值，我们可以使用切片符「:」，和Python中使用方法一样：
my_array[2:6] 
array([2, 3, 4, 5])
my_array[:5]
array([0, 1, 2, 3, 4])
 my_array[5:]
array([ 5,  6,  7,  8,  9, 10, 11, 12])
同样也可以通过使用 [ ][ ] 或 [,] 在二维数组中选择元素。
现在使用 [ ][ ] 从下面的二维数组中抓取出值「60」：
two_d_arr = np.array([[10,20,30], [40,50,60], [70,80,90]])
two_d_arr[1][2] #抓取第二行第三列 
使用 [,] 从上面的二维数组中抓取出值「20」：
two_d_arr[0,1] #抓取第二行第三列

也可以用切片符抓取二维数组的子部分。使用下面的操作从数组中抓取一些元素：


two_d_arr[:1, :2]           # This returns [[10, 20]]
two_d_arr[:2, 1:]           # This returns ([[20, 30], [50, 60]])
two_d_arr[:2, :2]           #This returns ([[10, 20], [40, 50]])

我们还可以索引一整行或一整列。只需使用索引数字即可抓取任意一行：


two_d_arr[0]    #This grabs row 0 of the array ([10, 20, 30])
two_d_arr[:2] #This grabs everything before row 2 ([[10, 20, 30], [40, 50, 60]])

还可以使用 &、|、<、> 和 == 运算符对数组执行条件选择和逻辑选择，从而对比数组中的值和给定值：


new_arr = np.arange(5,15)
new_arr > 10 #This returns TRUE where the elements are greater than 10 [False, False, False, False, False, False,  True,  True,  True, True]

现在我们可以输出符合上述条件的元素：


bool_arr = new_arr > 10
new_arr[bool_arr]  #This returns elements greater than 10 [11, 12, 13, 14]
new_arr[new_arr>10] #A shorter way to do what we have just done

组合使用条件运算符和逻辑运算符，我们可以得到值大于 6 小于 10 的元素：


new_arr[(new_arr>6) & (new_arr<10)]

预期结果为：([7, 8, 9])


广播机制


广播机制是一种快速改变 NumPy 数组中的值的方式。


my_array[0:3] = 50
#Result is: 
[50, 50, 50, 3, 4,  5,  6,  7,  8,  9, 10]

在这个例子中，我们将索引为 0 到 3 的元素的初始值改为 50。


对 NumPy 数组执行数学运算


arr = np.arange(1,11)
arr * arr              #Multiplies each element by itself 
arr - arr              #Subtracts each element from itself
arr + arr              #Adds each element to itself
arr / arr              #Divides each element by itself

我们还可以对数组执行标量运算，NumPy 通过广播机制使其成为可能： 


arr + 50              #This adds 50 to every element in that array

NumPy 还允许在数组上执行通用函数，如平方根函数、指数函数和三角函数等。


np.sqrt(arr)     #Returns the square root of each element 
np.exp(arr)     #Returns the exponentials of each element
np.sin(arr)     #Returns the sin of each element
np.cos(arr)     #Returns the cosine of each element
np.log(arr)     #Returns the logarithm of each element
np.sum(arr)     #Returns the sum total of elements in the array
np.std(arr)     #Returns the standard deviation of in the array

我们还可以在二维数组中抓取行或列的总和：


mat = np.arange(1,26).reshape(5,5)
mat.sum()         #Returns the sum of all the values in mat
mat.sum(axis=0)   #Returns the sum of all the columns in mat
mat.sum(axis=1)   #Returns the sum of all the rows in mat


```
