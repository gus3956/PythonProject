import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 원본 데이터
dachshund_length = [77, 78, 85, 83, 73, 77, 73, 80]
dachshund_height = [25, 28, 29, 30, 21, 22, 17, 35]

samoyd_length = [75, 77, 86, 86, 79, 83, 83, 88]
samoyd_height = [56, 57, 50, 53, 60, 53, 49, 60]

# 평균 계산
dachshund_length_mean = np.mean(dachshund_length)
dachshund_height_mean = np.mean(dachshund_height)

samoyd_length_mean = np.mean(samoyd_length)
samoyd_height_mean = np.mean(samoyd_height)

# 새로운 데이터 생성 (200마리)
new_dachshund_length = np.random.normal(dachshund_length_mean, 10.0, 200)
new_dachshund_height = np.random.normal(dachshund_height_mean, 10.0, 200)

new_samoyed_length = np.random.normal(samoyd_length_mean, 10.0, 200)
new_samoyed_height = np.random.normal(samoyd_height_mean, 10.0, 200)

# 2차원 배열로 합치기
new_dachshund_data = np.column_stack((new_dachshund_length, new_dachshund_height))
new_samoyed_data = np.column_stack((new_samoyed_length, new_samoyed_height))


# 라벨 생성
dachshund_labels = np.zeros(len(new_dachshund_data))
samoyed_labels = np.ones(len(new_samoyed_data))

# 학습 데이터 병합
dogs = np.concatenate((new_dachshund_data, new_samoyed_data), axis=0)
labels = np.concatenate((dachshund_labels, samoyed_labels), axis=0)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(dogs, labels, test_size=0.2, random_state=42)

# 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# 테스트 정확도 출력
print(f"테스트 정확도: {knn.score(X_test, y_test)}")

# KNN 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(dogs, labels)
print(f"학습 정확도: {knn.score(X=dogs, y = labels)}")

# 무작위로 추출한 400마리 KNN 그래프
plt.scatter(new_dachshund_length, new_dachshund_height, c='c', marker='.')
plt.scatter(new_samoyed_length,new_samoyed_height, c = 'b', marker= '*')
plt.xlabel("Height")
plt.ylabel("Length")

plt.legend(["Dachshund", "Samoyed"],loc='upper left')
plt.show()

# 무작위 5마리 추출
indices = np.random.choice(len(dogs), 5, replace=False)
unknown_dogs = dogs[indices]

# 예측
predictions = knn.predict(unknown_dogs)


# 무작위 5마리 그래프
plt.scatter(unknown_dogs[:, 0], unknown_dogs[:, 1], c='r', marker='p')  # 5마리
plt.xlabel("Height")
plt.ylabel("Length")
plt.legend(["Unknown dogs"],loc='upper left')
plt.show()

# 결과 출력
dog_classes = {0: "닥스훈트", 1: "사모예드"}
for i, dog in enumerate(unknown_dogs):
    print(f"{i+1}번 개: 길이 {dog[0]:.1f} / 키 {dog[1]:.1f} → {dog_classes[predictions[i]]}")
