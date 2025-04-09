import tensorflow as tf
from keras.src.datasets.mnist import load_data
import keras
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = load_data()
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))
print(X_train.shape)
print(X_test.shape)

X_train = X_train/ 255.0
X_test = X_test/255.0

model = keras.Sequential([], name = "CNN")

input_layer = keras.Input(shape = (28, 28, 1), name = "InputLayer")
model.add(input_layer)
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', name = "Conv2D_1"))
model.add(keras.layers.MaxPool2D(pool_size= (2,2), name = "Maxpool2D_1"))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', name = "Conv2D_2"))
model.add(keras.layers.MaxPool2D(pool_size= (2,2), name = "Maxpool2D_2"))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', name = "Conv2D_3"))
model.add(keras.layers.MaxPool2D(pool_size= (2,2), name = "Maxpool2D_3"))

model.add(keras.layers.Flatten()) # 여기부터 DNN
model.add(keras.layers.Dense(units= 64, activation= 'relu',name="HiddenLayer1"))
model.add(keras.layers.Dense(units= 10, activation= 'softmax',name="OutputLayer1"))

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x= X_train, y = y_train, epochs= 1000, verbose= 'auto')
print(f'예측 정확도 : {model.evaluate(x = X_test, y = y_test)}')

model.save("2025_03_27_CNN.keras")

good_model = keras.models.load_model('2025_03_27_CNN.keras')

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

import cv2 as cv
img_list = ['number_1.png','number_2.png','number_3.png']
def image_procession(img_path):
     original =cv.imread(img_path, cv.IMREAD_GRAYSCALE)
     image= cv .resize(original, (28 , 28)) # 이미지를 사이즈 28 x 28
     image = 255 - image
     image = image.astype('float32')
     image = image.reshape(1, 28, 28 , 1) # 평탄화
     image = image / 255.0 # 예측할 이미지

for li in img_list:
    image = image_preprocess(li)
    predict_image = good_model.predict(image)
    print(f'그림 이미지 값은 : {predict_image}')
    print(f"추정된 숫자는 : {predict_image.argmax()}")

# 포트폴리오: 필기체 3개 던져주고 맞췄는지 확인, 그리고 loss accuracy 그래프 만들기