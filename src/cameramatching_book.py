import cv2

# 1. 캠 열기
cap = cv2.VideoCapture(0)

while True:
    # 2. 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 화면에 출력
    cv2.imshow('Camera', frame)

    # 4. 키 입력 대기: 스페이스바 누르면 캡처
    key = cv2.waitKey(1)
    if key == 32:  # 스페이스바
        break
    elif key == 27:  # ESC 누르면 그냥 종료
        frame = None
        break

# 5. 자원 정리
cap.release()
cv2.destroyAllWindows()
