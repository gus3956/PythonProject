import serial, time

bt = serial.Serial('COM3', 9600)  # ✅ USB 연결 (아두이노 메가)
time.sleep(2)

while True:
    bt.write(b'F\n')  # USB 통해 명령 전송 (아두이노 메가 → HC-05 → HC-06 바이봇)
    print("파이썬이 전송한 명령: F")
    time.sleep(1)

    if bt.in_waiting:
        response = bt.readline().decode().strip()
        print("바이봇으로부터 받은 데이터:", response)
