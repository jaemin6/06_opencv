# 📸 ORB 특징점 기반 이미지 검색 (캠 촬영 + ROI 매칭)

## ✅ 개요
- 사용자가 **웹캠으로 직접 사진을 찍고**, 관심영역(ROI)을 선택하면
- ORB 특징점 기반으로 **로컬 이미지들과 매칭**하여 가장 비슷한 이미지를 자동으로 찾아줍니다.
- 검색 시간도 함께 출력됩니다. ⏱️

---

## 🧪 사용 기술
- OpenCV (cv2)
- ORB (Oriented FAST and Rotated BRIEF)
- BFMatcher (Hamming 거리 기반 매칭)
- ROI (관심 영역) 선택
- 시간 측정 (`time.time()`)

---

## 🎮 사용 방법
1. 프로그램 실행 시 카메라가 켜짐
2. `스페이스 바`를 누르면 현재 화면을 촬영하고 ROI를 선택
3. 선택된 ROI 이미지를 ORB 특징점으로 분석하여
4. `../img/book*.jpg` 이미지들과 매칭
5. 가장 유사한 이미지를 화면에 출력함
6. `ESC` 키를 누르면 전체 프로그램 종료

---

## 💻 전체 코드 (주석 포함)

```python
import cv2
import numpy as np
import glob
import time  # 🔹 시간 측정용 모듈

# ORB 디텍터 (특징점 최대 1000개)
orb = cv2.ORB_create(1000)

# BFMatcher 생성 (해밍 거리 기반, 교차검증 사용)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 카메라 연결
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 종료
        break

    elif key == ord(' '):  # 스페이스: ROI 선택 후 검색
        x, y, w, h = cv2.selectROI("Camera", frame, False)
        if w and h:
            roi = frame[y:y+h, x:x+w]                      # ROI 추출
            kp1, des1 = orb.detectAndCompute(roi, None)   # ROI 특징 추출

            best_match_path = None
            best_match_count = 0
            img_paths = sorted(glob.glob("../img/book*.jpg"))

            start_time = time.time()  # ⏱️ 시작 시간

            for path in img_paths:
                db_img = cv2.imread(path)
                kp2, des2 = orb.detectAndCompute(db_img, None)

                if des1 is None or des2 is None:
                    continue

                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < 50]

                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                    best_match_path = path

            end_time = time.time()  # ⏱️ 종료 시간
            elapsed_time = end_time - start_time

            print(f"검색 시간: {elapsed_time:.2f}초")

            if best_match_path:
                best_img = cv2.imread(best_match_path)
                cv2.imshow("Best Match", best_img)
                print(f"Best match: {best_match_path} with {best_match_count} good matches")

# 종료 처리
cap.release()
cv2.destroyAllWindows()
