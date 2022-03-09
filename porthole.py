import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import json
import requests

#데이터셋 가공

IMG_SIZE=64 
IMG_SET="C:/Users/82105/Desktop/datase/" 
Class=["Road","Pothole"]
VALUE=[]
LABLE=[]


def Make_DataSetformat():
    for category in Class:
        path=os.path.join(IMG_SET,category)
        lable=Class.index(category)
        for img in os.listdir(path):
            try:
                img=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
                VALUE.append(img)
                LABLE.append(lable)
                
            except Exception as e:
                pass

Make_DataSetformat() 
VALUE=np.array(VALUE).reshape(-1,IMG_SIZE,IMG_SIZE,1) 
LABLE=np.array(LABLE)
LABLE=pd.get_dummies(LABLE)
VALUE=VALUE/255.0
random.shuffle(Data_Set)

#모델 구현 및 학습
Q=tf.keras.Input(shape=[64,64,1])
H=tf.keras.layers.Conv2D(6,kernel_size=5,padding='same',activation='relu')(Q)
H=tf.keras.layers.MaxPool2D()(H)
H=tf.keras.layers.Conv2D(16,kernel_size=5,activation='relu')(H)
H=tf.keras.layers.MaxPool2D()(H)
H=tf.keras.layers.Flatten()(H)
H=tf.keras.layers.Dense(120,activation='relu')(H)
H=tf.keras.layers.Dense(84,activation='relu')(H)
W=tf.keras.layers.Dense(2,activation='sigmoid')(H)

model=tf.keras.models.Model(Q,W)
model.compile(loss='categorical_crossentropy',metrics='accuracy')

model.fit(VALUE,LABLE,epochs=100)
model.save_weights('PotHole_Weight')


url='http://ip-api.com/json'
model.save_weights('PotHole_Weight')
cap = cv2.VideoCapture(0)
status="시작"
    
while (True):
    # cam에 인식되는 이미지 캡처
    
    data=requests.get(url)
    locate=data.json()

    ret, img = cap.read()

    if ret == False:  # 촬영 실패시 지속
        continue
    if cv2.waitKey(1) & 0xFF == 27:  # Esc키를 사용해 cam 화면 종료
        break
    
    # 캡처 이미지 변환
    imgg = cv2.resize(img, (64, 64))
    imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    imgg = np.expand_dims(imgg, axis=0)
    imgg = imgg / 255.0
    Predictions = model.predict(imgg)

    # cam화면에 표시 될 폰트 기본 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, status, (10,40), font, 1, (255, 0, 0), 2)
    
    if Predictions[0][0] > 0.5:  # 안전한 도로일 경우 CAM에 표시 할 내용

        A = str(int(Predictions[0][0] * 100))
        status = A + "% SAFE  "+"lat:"+str(locate['lat'])+", lon:"+str(locate['lon'])  # "얼마의 확률로 SAFE"
        org = (300, 50)
        size, BaseLine = cv2.getTextSize(status, font, 1, 2)
        img = cv2.rectangle(img, (10, 10), (635, 470), (0, 255, 0), 3)  # 녹색 사각형 출력

    else:  # PotHole존재하는 도로일 경우  CAM에 표시 할 내용

        A = str(int(Predictions[0][1] * 100))
        status = A + "% POTHOLE "+"lat:"+str(locate['lat'])+", lon:"+ str(locate['lon'])# "얼마의 확률로 POTHOLE"
        org = (300, 50)
        size, BaseLine = cv2.getTextSize(status, font, 1, 2)
        img = cv2.rectangle(img, (10, 10), (635, 470), (0, 0, 255), 3)  # 적색 사각형 출력

    cv2.imshow('rectangle', img)  # 설정된 폰트와 캡처 이미지 CAM화면에 출력

# CAM자원 반납
cap.release()
cv2.destroyAllWindows()







