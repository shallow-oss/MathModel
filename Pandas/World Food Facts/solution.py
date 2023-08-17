import pandas as pd
import numpy as np

food = pd.read_csv(
    r'C:\Users\shining3d\Desktop\en.openfoodfacts.org.products.tsv', sep='\t')
# 显示前五条数据
print(food.head())
# 显示数据的形态
print(food.shape)
print(food.shape[0])
# 显示数据的信息
print(food.info())
# 显示标签
print(food.columns)
print(food.columns[104])
print(food.dtypes['-glucose_100g'])
# 显示数据如何索引
print(food.index)
print(food.values[18][7])
