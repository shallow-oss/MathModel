import pandas as pd
import numpy as np

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'

chipo = pd.read_csv(url, sep='\t')

print(chipo.head(10))
print(chipo.shape[0])
print(chipo.info())
print(chipo.shape[1])
print(chipo.columns)
print(chipo.index)

c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
print(c.head())

c = chipo.groupby('choice_description')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
print(c.head())
