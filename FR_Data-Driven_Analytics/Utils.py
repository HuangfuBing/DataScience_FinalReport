import pandas as pd
import numpy as np
import jieba
import re
import wordcloud
import imageio
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from snownlp import SnowNLP
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def suffixCheck(str:str):
    length = len(str)
    pattern = '_x0000_'
    if(str[length-7:length] == pattern):
        return True
    return False

def devide(df:pd.DataFrame):
    words = []
    for i, row in df.iterrows():
        word = jieba.cut(row['描述'])
        result = ' '.join(word)
        words.append(result)
    print(words)
    return words

def train(df, words):
    #Extract features
    vect = CountVectorizer()
    X = vect.fit_transform(words)
    X = X.toarray()
    #Unique
    words_bag = vect.vocabulary_
    y = df['情感类别']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=1)
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mlp, file)

def getEmotion(df):
    comments = df['描述']
    values = [SnowNLP(i).sentiments for i in comments]
    tend = []
    for i in values:
        if(i>0.5):
            tend.append(1)
        else:
            tend.append(0)
    df['情感类别'] = tend
    df['EmotionScore'] = values
    return df

def getRate(df):
    values = df['EmotionScore']
    print(type(values))
    good = 0
    bad = 0
    for i in values :
        if(i>0.5):
            good +=1
        else:
            bad +=1
    y = values.values
    print('好评率',(good/(good+bad)*100, '%'))
    plt.rc('font', family='serif', size = 10)
    plt.plot(y, marker='o', mec='r', mfc='w', label=u'EmotionScore')
    plt.xlabel('User')
    plt.ylabel('EmotionScore')
    plt.legend()
    plt.title('情感分析得分', family='SimHei', size = 14, color='blue')
    plt.show()

def transform(df):
    values = df['描述'].values
    txt_file = open('comments.txt', 'w', encoding='utf-8')
    for i in values:
        txt_file.write(i + ' ')

def drawCloud(path:str):
    stopwords = set()
    content = [line.strip() for line in open('./cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
    stopwords.update(content)

    with open(path, encoding='utf-8') as f:
        data = f.read()
    ChineseData = re.findall('[\u4e00-\u9fa5]+', data, re.S)
    ChineseData = '/'.join(ChineseData)

    seg_list_exact = jieba.cut(ChineseData, cut_all=True)
    result_list = []
    for word in seg_list_exact:
        result_list.append(word)

    py = imageio.imread('./phone.png')

    wc = wordcloud.WordCloud(
        width = 1000,
        height = 700,
        background_color='white',
        mask=py,
        scale=15,
        font_path='./STKAITI.TTF',
        stopwords= stopwords
    )

    data = ' '.join(result_list[:500])
    wc.generate(data)

    wc.to_file('./output.png')

