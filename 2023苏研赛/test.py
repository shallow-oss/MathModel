import pulp

# 数据
num_devices = 48
num_stations = 9
num_operations = 9  # 假设有9个工序
epsilon = 1e-4
num_tracks = 2
# 设备测试时间数据
data = [
    [0, 0, 0, 0, 0, 0, 25, 60, 40],
    [0, 0, 0, 0, 0, 0, 25, 60, 40],
    [0, 0, 0, 0, 0, 0, 25, 60, 40],
    [0, 0, 0, 0, 0, 0, 25, 60, 40],
    [0, 0, 0, 0, 0, 0, 25, 60, 40],
    [0, 0, 0, 0, 0, 0, 25, 60, 40],
    [0, 0, 0, 0, 0, 0, 30, 60, 50],
    [0, 0, 0, 0, 0, 0, 30, 60, 50],
    [0, 0, 0, 0, 0, 0, 30, 60, 50],
    [0, 0, 0, 0, 0, 0, 30, 60, 50],
    [0, 0, 0, 0, 0, 0, 30, 60, 50],
    [0, 0, 0, 0, 0, 0, 30, 60, 50],
    [0, 0, 0, 25, 120, 45, 35, 70, 45],
    [0, 0, 0, 25, 120, 45, 35, 70, 45],
    [0, 0, 0, 25, 120, 45, 35, 70, 45],
    [0, 0, 0, 25, 120, 45, 35, 70, 45],
    [0, 0, 0, 25, 120, 45, 35, 70, 45],
    [0, 0, 0, 25, 120, 45, 35, 70, 45],
    [0, 0, 0, 30, 105, 45, 40, 70, 55],
    [0, 0, 0, 30, 105, 45, 40, 70, 55],
    [0, 0, 0, 30, 105, 45, 40, 70, 55],
    [0, 0, 0, 30, 105, 45, 40, 70, 55],
    [0, 0, 0, 30, 105, 45, 40, 70, 55],
    [0, 0, 0, 30, 105, 45, 40, 70, 55],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [30, 65, 90, 40, 180, 60, 45, 75, 60],
    [35, 70, 90, 35, 200, 60, 50, 80, 65],
    [35, 70, 90, 35, 200, 60, 50, 80, 65],
    [35, 70, 90, 35, 200, 60, 50, 80, 65],
    [35, 70, 90, 35, 200, 60, 50, 80, 65],
    [35, 70, 90, 35, 200, 60, 50, 80, 65],
    [35, 70, 90, 35, 200, 60, 50, 80, 65],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
    [40, 80, 90, 35, 200, 60, 50, 80, 70],
]

t = {
    i: {
        j: {
            k: data[i][k]
            for k in range(num_operations)
        }
        for j in range(num_stations)
    }
    for i in range(num_devices)
}

# 创建一个线性规划问题实例
problem = pulp.LpProblem("Device_Testing", pulp.LpMinimize)

# 决策变量
x = pulp.LpVariable.dicts("x", (range(num_devices), range(
    num_stations), range(num_operations), range(num_tracks)), cat="Binary")
T = pulp.LpVariable("T", lowBound=0)

# 目标函数
problem += T

# 约束条件 1
for i in range(num_devices):
    for k in range(num_operations):
        problem += pulp.lpSum(x[i][j][k][l] for j in range(num_stations)
                              for l in range(num_tracks)) == 1

# 约束条件 2
for i in range(num_devices):
    for k in range(num_operations - 1):
        for k_prime in range(k + 1, num_operations):
            problem += pulp.lpSum(t[i][j][k] * x[i][j][k][l] for j in range(num_stations) for l in range(num_tracks)) <= \
                pulp.lpSum(t[i][j][k_prime] * x[i][j][k_prime][l]
                           for j in range(num_stations) for l in range(num_tracks)) - epsilon

# 约束条件 3
for j in range(num_stations):
    for l in range(num_tracks):
        problem += pulp.lpSum(x[i][j][k][l] for i in range(num_devices)
                              for k in range(num_operations)) <= 1

# 约束条件 4
for i in range(num_devices):
    for j in range(num_stations):
        for k in range(num_operations):
            problem += pulp.lpSum(t[i][j][k] * x[i][j][k][l]
                                  for l in range(num_tracks)) <= T

# 求解问题
problem.solve()

# 输出结果
print("Status:", pulp.LpStatus[problem.status])
print("Minimum testing time:", pulp.value(T))
