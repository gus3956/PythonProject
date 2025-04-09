import serial
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow.keras.models import load_model
from datetime import datetime
import time
import csv

# 🔹 모델 및 인코더 로드
cmd_model = load_model('cmd_model.keras') # cmd 모델 로드
rgb_model = load_model('rgb_model.keras') # rgb 모델 로드

with open('scaler.pkl', 'rb') as f: # 센서 데이터를 정규화 ( 0 ~ 1) 변환
    scaler = pickle.load(f)
with open('01.pkl', 'rb') as f: # 딥러닝 모델의 숫자 출력을 명령어 문자열로 되돌림 ( 0 -> F, 1 -> BF)
    cmd_encoder = pickle.load(f)

# 🔹 모델 워밍업
dummy = scaler.transform([[0, 0, 0, 0]]) # 초기 지연 줄이기
cmd_model.predict(dummy, verbose=0) # 예측중 출력 생략
rgb_model.predict(dummy, verbose=0) # 예측중 출력 생략

# 🔹 시리얼 연결
bt = serial.Serial('COM3', 9600, timeout=1)
bt.flushInput()
time.sleep(2)
print("✅ 연결 완료")

# ===== 맵 설정 =====
map_size = 60 # 맵 사이즈
robot_pos = [map_size // 2, map_size // 2] # 가로 세로 사이즈
direction = 'N' # 처음은 N
rotate_left = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}  # 로봇이 왼쪽 돌았을때 바이봇 바라보는 기준
rotate_right = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'} # 로봇이 오른쪽 돌았을때 바이봇 바라보는 기준
path = [] # 맵 이동 경로
visited = set() # 이미 방문한 위치
obstacles = set() # 장애물 위치
rgb_positions = [] # RGB 켰던 장소
cmd_history = [] # 명령어 기록
last_prediction_time = time.time() # 예측 시간 간격 조절
last_rgb_status = "OFF" # RGB ON/OFF 상태 기억

# ===== CSV 저장 =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = open(f"log_{timestamp}.csv", 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['CDS', 'distance', 'leftValue', 'rightValue', 'command', 'RGB'])

# ===== 유틸 함수 =====
def move_forward():
    if direction == 'N': robot_pos[1] += 1
    elif direction == 'S': robot_pos[1] -= 1
    elif direction == 'E': robot_pos[0] += 1
    elif direction == 'W': robot_pos[0] -= 1

def move_back():
    if direction == 'N': robot_pos[1] -= 1
    elif direction == 'S': robot_pos[1] += 1
    elif direction == 'E': robot_pos[0] -= 1
    elif direction == 'W': robot_pos[0] += 1

def update_position(cmd): # 맵 위치 기록 업데이트 함수
    global direction
    if cmd == 'BL':
        move_back()
        direction = rotate_left[direction]
    elif cmd == 'BR':
        move_back()
        direction = rotate_right[direction]
    elif cmd == 'B':
        move_back()
    else:
        if 'F' in cmd:
            move_forward()
        if 'L' in cmd:
            direction = rotate_left[direction]
        if 'R' in cmd:
            direction = rotate_right[direction]
    visited.add(tuple(robot_pos))
    path.append(tuple(robot_pos))

def is_obstacle_ahead():
    dx, dy = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}[direction]
    return (robot_pos[0]+dx, robot_pos[1]+dy) in obstacles

def estimate_obstacle(distance): # 장애물
    dx, dy = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}[direction]
    obs_x = robot_pos[0] + dx * distance // 5 # 1칸당 거리 약 5cm 가정해서 그만큼 떨어진 칸 계산
    obs_y = robot_pos[1] + dy * distance // 5 # 1칸당 거리 약 5cm 가정해서 그만큼 떨어진 칸 계산
    if 0 <= obs_x < map_size and 0 <= obs_y < map_size: # 맵 범위를 벗어나지 않으면 장애물 집합 추가
        obstacles.add((obs_x, obs_y))

def get_unvisited_direction(): # 아직 방문하지 않았고 장애물도 없는 방향을 찾아 리턴
    deltas = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
    for d in ['N', 'E', 'S', 'W']: # 시계 방향으로 하나씩 방향 확인
        dx, dy = deltas[d]
        nx, ny = robot_pos[0]+dx, robot_pos[1]+dy
        if 0 <= nx < map_size and 0 <= ny < map_size: # 방문한 적 없고 장애물도 아니면 d 방향으로 반환
            if (nx, ny) not in visited and (nx, ny) not in obstacles:
                return d
    return None

def get_turn_command(target_dir):
    if target_dir == direction: return 'F'
    elif rotate_left[direction] == target_dir: return 'L'
    elif rotate_right[direction] == target_dir: return 'R'
    elif rotate_left[rotate_left[direction]] == target_dir: return 'BL'
    elif rotate_right[rotate_right[direction]] == target_dir: return 'BR'
    else: return 'STOP'

# ===== 애니메이션 루프 =====
def animate(i): # 주행 중 센서 읽기 + 예측 + 명령 전송 + 맵 시각화
    global last_prediction_time, last_rgb_status # 전역 변수로 이전 상태 기억

    while bt.in_waiting: # 블루투스에 수신 데이터가 있으면 읽기
        line = bt.readline().decode('utf-8', errors='ignore').strip() # 한줄 전체를 받아 문자열로 반환
        if "distance" in line and "CDS" in line: #(ex: CDS: 320\t distance: 70\t leftvalue: 1\t rightvalue: 1)
            try:
                parts = line.split("\t")
                cds = int(parts[0].split(":")[1].strip())
                distance = int(parts[1].split(":")[1].strip())
                left = int(parts[2].split(":")[1].strip())
                right = int(parts[3].split(":")[1].strip())

                if time.time() - last_prediction_time >= 0.5:  # 예측주기: 0.5초
                    X = scaler.transform([[distance, left, right, cds]]) # 입력을 학습에 맞게 정규화, 이동 명령,RGB 예측
                    cmd_pred = cmd_model.predict(X, verbose=0)
                    rgb_pred = rgb_model.predict(X, verbose=0)

                    cmd_label = cmd_encoder.inverse_transform(np.argmax(cmd_pred, axis=1))[0] # 명령어 문자열
                    rgb_label = 'ON' if rgb_pred[0][0] > 0.5 else 'OFF' #RGB 확률 값이 0.5 초과면 ON

                    if cmd_label == 'F' and is_obstacle_ahead(): # 정면에 장애물 있을경우 방문하지 않은 다른 방향으로 회피
                        alt_dir = get_unvisited_direction()
                        cmd_label = get_turn_command(alt_dir) if alt_dir else 'STOP' # 회피할 방향 없으면 STOP

                    bt.write((cmd_label + '\n').encode('ascii'))
                    time.sleep(0.05) # 모터 명령과 RGB 명령 사이 간섭 방지
                    bt.write((f"RGB_{rgb_label}\n").encode('ascii'))

                    if distance < 12: # 가까운 거리일경우 장애물로 간주 기록
                        estimate_obstacle(distance)
                    update_position(cmd_label)

                    csv_writer.writerow([cds, distance, left, right, cmd_label, rgb_label])
                    if rgb_label == 'ON':
                        rgb_positions.append(tuple(robot_pos))
                    last_prediction_time = time.time() # 마지막 예측 시간, 마지막 RGB 상태, 명령 히스토리 저장
                    last_rgb_status = rgb_label
                    cmd_history.append(cmd_label)

            except Exception as e:
                print("❌ 예외 발생:", e)

    # ===== 맵 시각화 =====
    ax.clear()
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_title(f"Pos: {robot_pos} | Dir: {direction} | RGB: {last_rgb_status}") # 제목에 현재 위치 방향 RGB 표시
    ax.set_facecolor('#222' if last_rgb_status == 'ON' else 'white') # RGB 가 켜져 있으면 어두운 배경 꺼지면 흰색

    for (x, y) in obstacles: # 장애물 검은색 네모 그림
        ax.plot(x+1, y+1, 'ks', markersize=18)
    if len(path) > 1: # 이동한 좌표들 연결해 선으로 그림
        xs, ys = zip(*path)
        ax.plot([x+0.5 for x in xs], [y+0.5 for y in ys], 'blue', linewidth=2)
    ax.plot(robot_pos[0]+0.5, robot_pos[1]+0.5, 'ro', markersize=16) # 현재 로봇 위치 빨간색 ro 표시
    for (x, y) in rgb_positions: # RGB 켜졌던 좌표는 노란 원 yo 그림
        ax.plot(x+0.5, y+0.5, 'yo', markeredgecolor='black', markersize=8)

    dx, dy = {'N': (0, 1.2), 'S': (0, -1.2), 'E': (1.2, 0), 'W': (-1.2, 0)}[direction] # 로봇이 어느 방향으로 보고 있는지 시각적으로 확인 가능
    ax.quiver(robot_pos[0]+0.5, robot_pos[1]+0.5, dx, dy, angles='xy', scale_units='xy', scale=1, color='red')

# ===== 실행 =====
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, animate, interval=400, cache_frame_data=False)
plt.show()
