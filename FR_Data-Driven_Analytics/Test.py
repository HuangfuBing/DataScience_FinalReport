import pandas as pd
import jieba

import Utils

df = pd.read_csv('./data.csv', encoding='utf-8').astype(str)
print(df.head())

