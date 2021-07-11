from picamera import PiCamera        #카메라를 사용하기 위한 모듈
from time import sleep, time         #파이썬 시간 제어 모듈
import RPi.GPIO as GPIO              #RPi 동작을 위한 모듈
from keras.models import load_model  #keras, 오픈소스 신경망 라이브러리
import cv2                           #이미지 처리용 라이브러리
import numpy as np                    #행렬계산용도
import time
print("모듈 처리 완료")
SWITCH_PIN = 16                #switch pin
SEGMENT_PINS = [2,3,4,5,6,7,8] #7segment pins(7개)

GPIO.setmode(GPIO.BCM)

#각 GPIO핀 설정
for segment in SEGMENT_PINS:
    GPIO.setup(segment, GPIO.OUT)
    GPIO.output(segment, GPIO.LOW)

#Common Cathode (HIGH -> ON, LOW -> OFF)
data = [[1, 1, 1, 1, 1, 1, 0],  # 0
        [0, 1, 1, 0, 0, 0, 0],  # 1
        [1, 1, 0, 1, 1, 0, 1],  # 2
        [1, 1, 1, 1, 0, 0, 1],  # 3
        [0, 1, 1, 0, 0, 1, 1],  # 4
        [1, 0, 1, 1, 0, 1, 1],  # 5
        [1, 0, 1, 1, 1, 1, 1],  # 6
        [1, 1, 1, 0, 0, 0, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 0, 0, 1, 1]]  # 9
no_input = 0 # 입력주기 계산하기 위한 변수
#내부 풀다운저항(안눌렀을떄 0, 눌렀을때 :1)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#손글씨 데이터베이스(MNIST handwritten digit database)를 바탕으로 학습된 모델링 불러오기 (MNIST_CNN_model.h5 에서)
model = load_model('/home/pi/MNIST/MNIST_CNN_model.h5') 

print("PIN 설정 완료")
print("스위치를 누르면 촬영 시작")
print("10초동안 입력이 없을시 종료됩니다.")
camera = PiCamera()

try:
    while True:
        val = GPIO.input(SWITCH_PIN) #SWITCH_PIN에서 입력값을 받는다 (HIGH : 1, LOW : 0)

        if val == 1:
            print("camera_start")	
                #카메라 설정             

            camera.start_preview() #카메라 촬영준비
            sleep(5)               #카메라 실행을 위한 준비시간

            camera.capture('/home/pi/iot/capture2.png')                           #카메라 촬영 후 /home/pi/capture.png 위치에 저장
            camera.stop_preview()                                                 #촬영 종료
            
            # 인공지능 학습 모델 처리
            my_img = cv2.imread('/home/pi/iot/capture2.png', cv2.IMREAD_GRAYSCALE) #저장된 이미지 읽어오기
            my_img = cv2.resize(255-my_img, (28, 28))                              #학습된 사이즈에 맞게 28*28크기로 변환
            test_my_img = my_img.flatten() / 255.0
            test_my_img = test_my_img.reshape((-1, 28, 28, 1))
            num = np.argmax(model.predict(test_my_img), axis=-1)                  # 모델에서 예측한 값을 num 변수 저장
            print(num[0])
            #인식한 num값을 7segment 로 출력
            for j in range(7):  
                GPIO.output(SEGMENT_PINS[j], data[num[0]][j])
            time.sleep(1)
            val = 0
            no_input = 0
        time.sleep(0.2)
        no_input +=1
        if no_input>50: #10초동안 입력이 없을시 반복문을 종료하고 종료
            print("10초 이상 입력이 없으므로 종료됩니다.")
            break


finally: #오작동시 안전한 프로그램 종료를 위해서 마지막에 사용되고 종료된다.
    GPIO.cleanup()
    print('cleanup and exit')

# 참고
# camera = PiCamera()
# camera.start_preview()
# sleep(5)
# camera.capture('/home/pi/capture.jpg')
# camera.stop_preview()

# import cv2
# # 이미지 읽기
# img = cv2.imread('capture.jpg', 1)

# # 이미지 화면에 표시
# cv2.imshow('Test Image', img)
# cv2.waitKey(0)
 
# # 이미지 다른 파일로 저장
# cv2.imwrite('captrue2.png', img)

# #내부 풀다운저항(안눌렀을떄 0, 눌렀을때 :1)
