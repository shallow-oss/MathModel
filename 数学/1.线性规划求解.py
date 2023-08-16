from scipy.optimize import linprog

# 定义线性规划问题的目标函数系数
c = [-2, -3, 5]

# 定义线性规划问题的不等式约束矩阵
A = [[-2, 5, -1],
     [1, 3, 1]]

# 定义线性规划问题的不等式约束右侧常数
b = [-10, 12]

# 定义线性规划问题的等式约束矩阵
A_eq = [[1, 1, 1]]

# 定义线性规划问题的等式约束右侧常数
b_eq = [7]

# 定义线性规划问题的变量边界
x_bounds = [(0, None), (0, None), (0, None)]

# 求解线性规划问题
result = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds)

# 打印求解结果
print("最优解：", result.x)
print("最优目标值：", -result.fun)
