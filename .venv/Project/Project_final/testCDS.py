import pandas as pd
import numpy as np

# 데이터 개수
num_samples = 1000

# 랜덤 데이터 생성
np.random.seed(42)
distance = np.random.randint(0, 30, size=num_samples)
leftValue = np.random.randint(0, 100, size=num_samples)
rightValue = np.random.randint(0, 100, size=num_samples)
CDS = np.random.randint(300, 700, size=num_samples)

# 명령 생성 로직
commands = []
rgb_status = []

# 명령 생성 로직 수정: 거리 5 이하일 경우 B, 그 외 BL 또는 BR 또는 F
for d, lv, rv, c in zip(distance, leftValue, rightValue, CDS):
    # CMD 결정
    if d <= 5:
        cmd = 'B'
    elif 5 < d < 12:
        cmd = 'BL' if lv < rv else 'BR'
    else:
        cmd = 'F'
    commands.append(cmd)

    # RGB 결정
    rgb = 'ON' if c < 520 else 'OFF'
    rgb_status.append(rgb)


# 데이터프레임 생성
data = pd.DataFrame({
    'distance': distance,
    'leftValue': leftValue,
    'rightValue': rightValue,
    'CDS': CDS,
    'command': commands,
    'RGB': rgb_status
})

# CSV 저장
data.to_csv('biobot_train_data02.csv', index=False)
