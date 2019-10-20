#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 19:23:11 2019

@author: anoop
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import xlsxwriter
nltk.download('stopwords')


data_train=pd.read_excel("/home/anoop/Downloads/Participants_Data_News_category/Data_Train.xlsx")

data_test=pd.read_excel("/home/anoop/Downloads/Participants_Data_News_category/Data_Test.xlsx")

data_train.info()

data_train.SECTION.value_counts()

data_train=data_train.reset_index(drop=True)

def clean(text):
    replace1=re.compile('[/(){}\[\]\|@,;]')
    replace2=re.compile('[^0-9a-z #+_]')
    stopword=set(stopwords.words('english'))
    text=text.lower()
    text=replace1.sub(' ',text)
    text=replace2.sub('',text)
    texts=""
    for w in text.split():
        if w not in stopword:
            texts=texts+" "+w
    return texts 

data_train['STORY']=data_train['STORY'].apply(clean)  
data_test["STORY"]=data_test["STORY"].apply(clean)    

max_num=50000
max_len=506
embedding=100
tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=max_num, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True)
tokenizer.fit_on_texts(data_train['STORY'].values)
word_index=tokenizer.word_index

X = tokenizer.texts_to_sequences(data_train['STORY'].values)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len)

test=tokenizer.texts_to_sequences(data_test['STORY'].values)
test= tf.keras.preprocessing.sequence.pad_sequences(test, maxlen=max_len)

             
#y=data_train['SECTION']
y=pd.get_dummies(data_train['SECTION']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.05, random_state = 42)

model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_num,embedding,input_length=X.shape[1]))
model.add(tf.keras.layers.SpatialDropout1D(0.2))
model.add(tf.keras.layers.LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(4,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=5,batch_size=64,validation_split=0.1,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,min_delta=0.000)])

pred=model.predict(test)

label=['0','1','2','3']
result=[]
for t in pred:
    result.append(label[np.argmax(t)])
    
wb=xlsxwriter.Workbook('/home/anoop/results.xlsx')
ws=wb.add_worksheet()

for i in range(len(result)):
    ws.write(i,0,result[i])

wb.close()






    


