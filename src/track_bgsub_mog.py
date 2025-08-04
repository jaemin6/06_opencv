# 배경 제거  
import numpy as np, cv2

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)

# 배경 제거 객체 생성
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# history : 과거 프레임의 개수, 배경을 학습하는 데 얼마나 많은 프레임을 기억할지
# varThreshold: 픽셀이 객체인 배경인지 구분하는 기준 값
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 배경 제거 마스크 계산
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('bgsub', fgmask)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()
