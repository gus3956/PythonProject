import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import csv
from datetime import datetime

# ======== 설정 ========
bt = serial.Serial('COM3', 9600)
time.sleep(2)

map_size = 60
robot_pos = [map_size // 2, map_size // 2]
direction = 'N'
obstacles = set()
path = []
rgb_positions = []  # ✅ RGB: ON 위치 저장용 리스트
cmd_history = []
last_sensor_data = {}
last_rgb_status = "OFF"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"sensor_log_{timestamp}.csv"
csv_file = open(log_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['CDS', 'distance', 'leftValue', 'rightValue', 'command', 'RGB'])

rotate_left = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
rotate_right = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}

# ======== 파싱 함수 ========
def parse_sensor_data(line):
    data = {}
    if "distance" in line and "CDS" in line:
        parts = line.split('\t')
        for part in parts:
            if ':' in part:
                key_value = part.split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    try:
                        data[key.strip()] = int(value.strip())
                    except ValueError:
                        pass
    return data

def parse_command(line):
    for cmd in ['BL', 'BR', 'F', 'B', 'L', 'R', 'S']:
        if f"CMD: {cmd}" in line:
            return cmd
    return None

def parse_rgb_status(line):
    if "RGB: ON" in line:
        return "ON"
    elif "RGB: OFF" in line:
        return "OFF"
    return ""

# ======== 맵 업데이트 ========
def update_position(cmd):
    global direction, robot_pos, path
    if 'F' in cmd:
        if direction == 'N': robot_pos[1] += 1
        elif direction == 'S': robot_pos[1] -= 1
        elif direction == 'E': robot_pos[0] += 1
        elif direction == 'W': robot_pos[0] -= 1
    if 'B' in cmd:
        if direction == 'N': robot_pos[1] -= 1
        elif direction == 'S': robot_pos[1] += 1
        elif direction == 'E': robot_pos[0] -= 1
        elif direction == 'W': robot_pos[0] += 1
    if 'L' in cmd:
        direction = rotate_left[direction]
    if 'R' in cmd:
        direction = rotate_right[direction]
    path.append(tuple(robot_pos))

def estimate_obstacle(distance):
    dx, dy = 0, 0
    if direction == 'N': dy = 1
    elif direction == 'S': dy = -1
    elif direction == 'E': dx = 1
    elif direction == 'W': dx = -1
    obs_x = robot_pos[0] + dx * distance // 5
    obs_y = robot_pos[1] + dy * distance // 5
    if 0 <= obs_x < map_size and 0 <= obs_y < map_size:
        obstacles.add((obs_x, obs_y))

# ======== 애니메이션 ========
def animate(i):
    global cmd_history, last_sensor_data, last_rgb_status

    while bt.in_waiting:
        raw = bt.readline().decode(errors='ignore').strip()
        lines = raw.split("CDS: ")
        lines = [f"CDS: {line.strip()}" for line in lines if line.strip()]

        for line in lines:
            print("🔵 수신된 라인:", line)
            last_sensor_data = parse_sensor_data(line)
            cmd = parse_command(line)
            rgb_status = parse_rgb_status(line)
            distance = last_sensor_data.get("distance", 0)

            if rgb_status:
                last_rgb_status = rgb_status

            if distance <= 0:
                continue

            if 0 < distance < 12:
                estimate_obstacle(min(distance, 5))

            if cmd:
                update_position(cmd)
                cmd_history.append(cmd)
                if len(cmd_history) > 10:
                    cmd_history = cmd_history[-10:]
                csv_writer.writerow([
                    last_sensor_data.get('CDS', 0),
                    last_sensor_data.get('distance', 0),
                    last_sensor_data.get('leftValue', 0),
                    last_sensor_data.get('rightValue', 0),
                    cmd,
                    last_rgb_status
                ])

                # ✅ 현재 위치가 RGB: ON이면 마킹
                if last_rgb_status == "ON":
                    rgb_positions.append(tuple(robot_pos))

    # ===== 시각화 =====
    ax.clear()

    if last_rgb_status == "ON":
        ax.set_facecolor('#222222')
        line_color = 'yellow'
    else:
        ax.set_facecolor('white')
        line_color = 'blue'

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_title(f"Robot Pos: {robot_pos} | Dir: {direction} | RGB: {last_rgb_status} | Last Cmd: {cmd_history[-1] if cmd_history else 'None'}")

    for (x, y) in obstacles:
        ax.plot(x + 0.5, y + 0.5, 'ks', markersize=15)

    if len(path) > 1:
        x_vals = [p[0] + 0.5 for p in path]
        y_vals = [p[1] + 0.5 for p in path]
        ax.plot(x_vals, y_vals, line_color, linewidth=2)

    ax.plot(robot_pos[0] + 0.5, robot_pos[1] + 0.5, 'ro', markersize=16)

    # ✅ RGB: ON 위치 마킹
    for (x, y) in rgb_positions:
        ax.plot(x + 0.5, y + 0.5, 'yo', markersize=12, markeredgecolor='black')  # 노란 원 + 테두리

    # 방향 표시
    dx, dy = 0, 0
    if direction == 'N': dy = 1.2
    elif direction == 'S': dy = -1.2
    elif direction == 'E': dx = 1.2
    elif direction == 'W': dx = -1.2
    ax.quiver(robot_pos[0] + 0.5, robot_pos[1] + 0.5, dx, dy,
              angles='xy', scale_units='xy', scale=1, color='red', width=0.02)

# ======== 실행 ========
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, animate, interval=500, cache_frame_data=False)
plt.show()
