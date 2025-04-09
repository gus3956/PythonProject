import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('biobot_train_data02.csv')

# 특징과 라벨 분리
X = data[['distance', 'leftValue', 'rightValue', 'CDS']]
y_cmd = data['command']
y_rgb = data['RGB']

# 특징 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 명령어 라벨 인코딩
cmd_encoder = LabelEncoder()
y_cmd_enc = cmd_encoder.fit_transform(y_cmd)
y_cmd_cat = to_categorical(y_cmd_enc)

# RGB 라벨 인코딩
rgb_encoder = LabelEncoder()
y_rgb_enc = rgb_encoder.fit_transform(y_rgb)

# 학습, 검증 데이터 분리
X_train, X_test, y_cmd_train, y_cmd_test, y_rgb_train, y_rgb_test = train_test_split(
    X_scaled, y_cmd_cat, y_rgb_enc, test_size=0.2, random_state=42
)

# 명령 분류 모델
import keras
import tensorflow as tf
cmd_model = keras.models.Sequential(name = "CMD")
input_layer = keras.Input(shape = (4, ), name = "Input_layer")
cmd_model.add(input_layer)
cmd_model.add(keras.layers.Dense(units = 64, activation= 'relu', name = 'Layer1'))
cmd_model.add(keras.layers.Dense(units = 64, activation= 'relu', name = 'Layer2'))
cmd_model.add(keras.layers.Dense(units = 32, activation= 'relu', name = 'Layer3')) # 은닉층 Dense 64 64 32
output_layer = keras.layers.Dense(units=  y_cmd_cat.shape[1], activation= 'sigmoid', name = 'Output') # 출력층 1
cmd_model.add(output_layer)
cmd_model.summary() # 표로 요약해준다

cmd_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
cmd_model.fit(X_train, y_cmd_train, epochs=1000, batch_size=32, validation_split=0.2)

# RGB 분류 모델
import keras
import tensorflow as tf
rgb_model = keras.models.Sequential(name = "RGB")
input_layer = keras.Input(shape = (4, ), name = "Input_layer")
rgb_model.add(input_layer)
rgb_model.add(keras.layers.Dense(units = 16, activation= 'relu', name = 'Layer1'))
rgb_model.add(keras.layers.Dense(units = 8, activation= 'relu', name = 'Layer2'))
output_layer = keras.layers.Dense(units=  1, activation= 'sigmoid', name = 'Output') # 출력층 1
rgb_model.add(output_layer)
rgb_model.summary() # 표로 요약해준다
rgb_model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
history = rgb_model.fit(X_train, y_rgb_train, epochs=1000, batch_size=32, validation_split=0.2)


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
plt.legend(['Loss', 'Test'])
plt.show()
# 모델 저장 (딥러닝 모델)
cmd_model.save('cmd_model.keras')
rgb_model.save('rgb_model.keras')

# scaler와 LabelEncoder 저장 (pickle)
import pickle

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('01.pkl', 'wb') as f:
    pickle.dump(cmd_encoder, f)

with open('rgb_encoder.pkl', 'wb') as f:
    pickle.dump(rgb_encoder, f)

print("✅ 모든 모델과 인코더 저장 완료")
print(f'예측 정확도 : {cmd_model.evaluate(x = X_test, y = y_cmd_test)}')
# 학습 이후 간단한 예측 테스트 코드
test_samples = [
    [10, 30, 70, 500],  # distance <12, left<right ➜ BL, RGB ON
    [15, 80, 20, 550],  # distance >=12 ➜ F, RGB OFF
    [8, 60, 40, 300] ,
    [5, 60, 40, 300]
]


# 스케일링 적용
test_samples_scaled = scaler.transform(test_samples)

# 예측
cmd_predictions = cmd_model.predict(test_samples_scaled)
rgb_predictions = rgb_model.predict(test_samples_scaled)

cmd_labels = cmd_encoder.inverse_transform(np.argmax(cmd_predictions, axis=1))
rgb_labels = ['ON' if pred > 0.5 else 'OFF' for pred in rgb_predictions]

# 결과 출력
for i, sample in enumerate(test_samples):
    print(f"입력: {sample} ➜ 예측 명령: {cmd_labels[i]}, RGB 상태: {rgb_labels[i]}")
