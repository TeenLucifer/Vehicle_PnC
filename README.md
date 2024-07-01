## 车辆模型
- [侧向运动学和动力学模型](https://blog.csdn.net/TeenLucifer/article/details/139275063?spm=1001.2014.3001.5501)
- [以后轴中心为原点的车辆运动学模型](https://blog.csdn.net/TeenLucifer/article/details/139887147?spm=1001.2014.3001.5501)

## 车辆控制
- [LQR](https://blog.csdn.net/TeenLucifer/article/details/139451886?spm=1001.2014.3001.5501)
- [MPC](https://blog.csdn.net/TeenLucifer/article/details/139887288?spm=1001.2014.3001.5501)
- [Pure Puersuit](https://blog.csdn.net/TeenLucifer/article/details/140097970?spm=1001.2014.3001.5501)
- [Stanley](https://blog.csdn.net/TeenLucifer/article/details/140097970?spm=1001.2014.3001.5501)
- [PID]()

## 其它用到的工具
### 矩阵求导
矩阵求导感觉规则比较混乱，学习规控还是以会用那几个常用就行了。用到的矩阵求导的知识主要是优化或者线性化时对系统状态或者对系统输入求偏导、求雅各比矩阵、求海信矩阵等。理解分子布局、分母布局和混合布局，会用就行，可以参考这篇博客。[矩阵求导](https://blog.csdn.net/TeenLucifer/article/details/139858158?spm=1001.2014.3001.5501)

### python求解二次规划问题
求解QP问题，可以用[OSQP求解器](https://osqp.org/docs/get_started/index.html)，也可以用[qpsolvers接口](https://pypi.org/project/qpsolvers/)。

用OSQP的话，对数据格式有严格要求，要传P, q, A, l, u四个参数，其中P和A要用scipy.sparse.csc_matrix格式的稀疏矩阵，q, l, u要用np.array格式的数据。表示如下二次规划问题：
$$
\begin{aligned}
    &\frac{1}{2}x^T P x + q^t x\\
    &s.t. \quad l \leq Ax \leq u
\end{aligned}
$$
此外，OSQP如果想要更新QP问题的参数，需要调用一系列update函数，比较麻烦。（试了下更新参数矩阵的函数，同样的参数到后面问题不可解，不知道为什么，用得不对？）下面是一个调用demo：

总得来说qpsolvers接口好用一点，一方面它后台的求解器可以选择，另一方面它把一些求解器封装好了直接可以调用函数，不需要创建对象。更新其中的参数矩阵也方便，相当于直接求解一个新问题，不需要更新求解器对象，避免了很多麻烦。所求解的二次规划问题形式为：
$$
\begin{aligned}
    &\frac{1}{2}x^T P x + q^T x\\
    &\begin{aligned}
        s.t. \quad &Gx \leq h\\
                   &Ax = b\\
                   &lb \leq x \leq ub
    \end{aligned}
\end{aligned}
$$
qpsolvers接口要传P, q, G, h, A, b, lb, ub八个参数，参数都可以用np.array格式的数据表示，或者P, q, A, G这四个矩阵也可以用scipy.sparse.csc_matrix格式的稀疏矩阵表示，能够加速计算。下面是一个调用demo：