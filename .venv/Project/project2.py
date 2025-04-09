import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

(X, y) = load_breast_cancer(return_X_y= True) # 특성과 라벨 분리
print(X[0])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size= 0.8, stratify= y, random_state= 42) # X는 훈련용 데이터 80% , y는 훈련,테스트 비율 일정, 랜덤 시드 42개


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # 정규화
scaler.fit(X_train) #X_train 값이 안바뀜
scaler.transform(X_train) # 값이 바뀜
scaler.fit(X_test)
scaler.transform(X_test)
print()
print(X_train[0])
print(X_train.shape)

import keras
import tensorflow as tf
model = keras.models.Sequential(name = "Predict_Cancer")
input_layer = keras.Input(shape = (30, ), name = "Input_layer")
model.add(input_layer)
model.add(keras.layers.Dense(units = 64, activation= 'relu', name = 'Layer1'))
model.add(keras.layers.Dense(units = 64, activation= 'relu', name = 'Layer2'))
model.add(keras.layers.Dense(units = 32, activation= 'relu', name = 'Layer3')) # 은닉층 Dense 64 64 32
output_layer = keras.layers.Dense(units= 1, activation= 'sigmoid', name = 'Output') # 출력층 1
model.add(output_layer)
model.summary() # 표로 요약해준다
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics=['accuracy']) #모델 학습 준비
history = model.fit(x = X_train, y = y_train, epochs = 10000, verbose= 'auto') # 10000번 학습

print(f'예측 정확도 : {model.evaluate(x = X_test, y = y_test)}')

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history.history['loss'], color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()
# 유방암 진단 포트폴리오 만들기 사용언어 : 텐서플로