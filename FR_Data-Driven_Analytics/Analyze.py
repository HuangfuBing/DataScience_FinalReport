import pandas as pd
import jieba

import Utils

df = pd.read_csv('./data.csv', encoding='utf-8').astype(str)

#Data preprocessing
df['类别'] = df['类别'].str.replace('_x0000_','')
df['描述'] = df['描述'].str.replace('_x0000_','')
df.to_csv('./data.csv', encoding='utf-8', index=False)

#Analysis
DeviceTypelist = ['T2x','Y33e','Y35+','Y77','Y77e','Y78','Y78+','Y78m','iQOO U5e','iQOO U5x','iQOO Z7','iQOO Z7x']

groups = df.groupby('机型')

T2x_df = groups.get_group('T2x')
Y33e_df = groups.get_group('Y33e')
Y35Plus_df = groups.get_group('Y35+')
Y77_df = groups.get_group('Y77')
Y77e_df = groups.get_group('Y77e')
Y78_df = groups.get_group('Y78')
Y78Plus_df = groups.get_group('Y78+')
Y78m_df = groups.get_group('Y78m')
iQ00U5e_df = groups.get_group('iQOO U5e')
iQ00u5X_df = groups.get_group('iQOO U5x')
iQ00Z7_df = groups.get_group('iQOO Z7')
iQ00Z7x_df = groups.get_group('iQOO Z7x')

DeviceTypeDataFrameList = [T2x_df,Y33e_df,Y35Plus_df,Y77_df,Y77e_df,Y78_df,Y78Plus_df,Y78m_df,iQ00U5e_df,iQ00u5X_df,iQ00Z7_df,iQ00Z7x_df]
for i in DeviceTypeDataFrameList:
    print(i)

#Emotion analyze
#Utils.getEmotion(iQ00Z7x_df)
#Utils.getRate(iQ00Z7x_df)

#Model training for prediction
#words = Utils.devide(df)
#Utils.train(df, words)

#WordCloud
Utils.transform(df)
Utils.drawCloud('./comments.txt')

