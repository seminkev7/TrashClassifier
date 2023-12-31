# -*- coding: utf-8 -*-
"""AI_DL_Multiclass.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AxcLMMgrZ92_K65g71mbZ_olxQpiTGwZ

# **Library Import**
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras

"""# **GPU**"""

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

"""# **Data Loading** """

base_dir ="C:/Users/tjxod/Downloads/garbage/garbage"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

target_size = (256, 256)
batch_size = 32

# 훈련 데이터 증강과 전처리를 위한 ImageDataGenerator 생성
train_datagen = ImageDataGenerator(
    featurewise_center=True,           # 특성별로 입력의 평균을 0으로 맞춤
    featurewise_std_normalization=True, # 특성별로 입력을 표준 편차로 나눔
    samplewise_center=True,            # 각 샘플의 평균을 0으로 맞춤
    horizontal_flip=True,               # 수평 방향으로 무작위로 이미지를 뒤집음
    vertical_flip=True,                 # 수직 방향으로 무작위로 이미지를 뒤집음
    rotation_range=15,                  # 무작위로 회전시킬 각도 범위
    zoom_range=0.1,                     # 무작위로 이미지를 확대/축소할 범위
    width_shift_range=0.15,             # 무작위로 이미지를 가로로 이동할 범위
    height_shift_range=0.15,            # 무작위로 이미지를 세로로 이동할 범위
    shear_range=0.1,                    # 무작위로 전단 변환을 적용할 범위
    fill_mode="nearest",                # 이미지를 변환할 때 채울 픽셀 값을 결정
    rescale=1./255,                     # 이미지의 픽셀 값을 0과 1 사이로 조정
    validation_split=0.2)               # 검증 데이터의 비율을 설정

# 검증 데이터를 위한 ImageDataGenerator 생성
validation_datagen = ImageDataGenerator(rescale=1./255)

# 테스트 데이터를 위한 ImageDataGenerator 생성
test_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 데이터 제네레이터 생성
train_generator = train_datagen.flow_from_directory(
    train_dir,
    classes=['cardboard','glass','metal','paper','plastic','trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# 검증 데이터 제네레이터 생성
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    classes=['cardboard','glass','metal','paper','plastic','trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# 테스트 데이터 제네레이터 생성
test_generator = test_datagen.flow_from_directory(
    test_dir,
    classes=['cardboard','glass','metal','paper','plastic','trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

"""# **Train 갯수 세기** """

import numpy as np

class_names = train_generator.class_indices
class_count = len(class_names)

# 클래스별 샘플 수를 저장할 딕셔너리 생성
class_sample_count = {class_name: 0 for class_name in class_names}

# train_generator를 모두 반복하며 클래스별 샘플 수 계산
for _ in range(len(train_generator)):
    _, labels = train_generator.next()
    for i in range(class_count):
        class_name = list(class_names.keys())[i]
        class_sample_count[class_name] += np.sum(labels[:, i])

# 클래스별 샘플 수 출력
for class_name, sample_count in class_sample_count.items():
    print(f"Class '{class_name}': {sample_count} samples")

"""# **Model Learning**"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Sequential 모델 생성
model = Sequential()

# 첫 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 두 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 세 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 네 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 레이어 추가
model.add(Flatten())

# Dropout 레이어 추가
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# 출력 레이어 추가
model.add(Dense(6, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 구조 요약 출력
model.summary()

# 모델 훈련
history = model.fit(train_generator, epochs=200, validation_data=validation_generator)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel(['epoch'])
plt.ylabel(['loss'])
plt.legend(['train','val'])
plt.show()

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel(['epoch'])
plt.ylabel(['loss'])
plt.legend(['train','val'])
plt.show()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 디렉토리 경로 설정
base_dir = "C:/Users/tjxod/Downloads/garbage/garbage"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

# 이미지 크기와 배치 크기 설정
target_size = (256, 256)
batch_size = 32

# 훈련 데이터에 적용할 데이터 증강 설정
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    fill_mode="nearest",
    rescale=1./255,
    validation_split=0.2
)

# 검증 데이터에 적용할 데이터 전처리 설정
validation_datagen = ImageDataGenerator(
    fill_mode="nearest",
    rescale=1./255
)

# 테스트 데이터에 적용할 데이터 전처리 설정
test_datagen = ImageDataGenerator(
    fill_mode="nearest",
    rescale=1./255
)

# 훈련 데이터 제네레이터 생성
train_generator = train_datagen.flow_from_directory(
    train_dir,
    classes=['cardboard','glass','metal','paper','plastic','trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# 검증 데이터 제네레이터 생성
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    classes=['cardboard','glass','metal','paper','plastic','trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 테스트 데이터 제네레이터 생성
test_generator = test_datagen.flow_from_directory(
    test_dir,
    classes=['cardboard','glass','metal','paper','plastic','trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Sequential 모델 생성
model = Sequential()

# 첫 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 두 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 세 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 네 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 레이어 추가
model.add(Flatten())

# Dropout 레이어 추가
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# 출력 레이어 추가
model.add(Dense(6, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 구조 요약
model.summary()

# 모델 훈련
history = model.fit(train_generator, epochs=200, validation_data=validation_generator)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel(['epoch'])
plt.ylabel(['loss'])
plt.legend(['train','val'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel(['epoch'])
plt.ylabel(['accuracy'])
plt.legend(['train','val'])
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 데이터 경로 설정
base_dir = "C:/Users/tjxod/Downloads/garbage/garbage"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

# 이미지 크기와 배치 크기 설정
target_size = (256, 256)
batch_size = 32

# 훈련 데이터에 적용할 데이터 증강 및 전처리 옵션 설정
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    samplewise_center=True,
    horizontal_flip=True, 
    vertical_flip=True,
    rotation_range=15, zoom_range=0.1,
    width_shift_range=0.15, height_shift_range=0.15,
    shear_range=0.1,
    fill_mode="nearest",
    rescale=1./255, 
    validation_split=0.2)

# 검증 데이터에 적용할 전처리 옵션 설정
validation_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    samplewise_center=True,
    horizontal_flip=True, 
    vertical_flip=True,
    rotation_range=15, zoom_range=0.1,
    width_shift_range=0.15, height_shift_range=0.15,
    shear_range=0.1,
    fill_mode="nearest",
    rescale=1./255)

# 테스트 데이터에 적용할 전처리 옵션 설정
test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    samplewise_center=True,
    horizontal_flip=True, 
    vertical_flip=True,
    rotation_range=15, zoom_range=0.1,
    width_shift_range=0.15, height_shift_range=0.15,
    shear_range=0.1,
    fill_mode="nearest",
    rescale=1./255)

# 훈련 데이터 제네레이터 생성
train_generator = train_datagen.flow_from_directory(
    train_dir,
    classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True) 

# 검증 데이터 제네레이터 생성
validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# 테스트 데이터 제네레이터 생성
test_generator = test_datagen.flow_from_directory(
    test_dir,
    classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Sequential 모델 생성
model = Sequential()

# 첫 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 두 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 세 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 네 번째 Convolutional 레이어 & Maxpooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 레이어 추가
model.add(Flatten())

# Dropout 레이어 추가
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# 출력 레이어 추가
model.add(Dense(6, activation='softmax'))  # Assuming 6 output classes

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 구조 출력
model.summary()

# 모델 훈련
history = model.fit(train_generator, epochs=200, validation_data=validation_generator)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel(['epoch'])
plt.ylabel(['loss'])
plt.legend(['train','val'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel(['epoch'])
plt.ylabel(['accuracy'])
plt.legend(['train','val'])
plt.show()

"""# **Rsnet**"""

from keras.applications import ResNet50

# ResNet50 모델 로드
model = ResNet50(include_top=True, weights=None, input_shape=(256, 256, 3), pooling='max', classes=6)

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# 모델 훈련
history = model.fit(train_generator, epochs=50, validation_data=validation_generator)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel(['epoch'])
plt.ylabel(['loss'])
plt.legend(['train','val'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel(['epoch'])
plt.ylabel(['accuracy'])
plt.legend(['train','val'])
plt.show()