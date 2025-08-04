"""
1. 캠이 열리면 스페이스바로 사진을 촬영
2. 엔터를 누르면 해당 경로에 있는 이미지에서 매칭되는 이미지를 찾음
3. esc를 누르면 찾은 이미지 창이 닫힘
4. 1번으로 돌아감
"""
import cv2
import numpy as np
import glob

# ORB 특징점 검출기 생성 (1000개 특징점까지)
orb = cv2.ORB_create(1000)

# BFMatcher 생성 (해밍 거리 사용, 교차 검증=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 카메라 열기 (0번 장치)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 너비 640 픽셀
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 높이 480 픽셀

while True:
    ret, frame = cap.read()   # 카메라에서 한 프레임 읽기
    if not ret:
        break                # 실패하면 종료

    cv2.imshow("Camera", frame)  # 원본 카메라 화면 보여주기
    key = cv2.waitKey(1) & 0xFF # 키 입력 받기

    if key == 27:  # ESC 키 누르면 종료
        break

    elif key == ord(' '):  # 스페이스 바 눌렀을 때
        # 사용자에게 ROI(관심영역) 선택 요청
        x, y, w, h = cv2.selectROI("Camera", frame, False)
        if w and h:
            roi = frame[y:y+h, x:x+w]         # 선택된 영역 이미지 자르기

            # ROI 이미지에서 ORB 특징점과 디스크립터 계산
            kp1, des1 = orb.detectAndCompute(roi, None)

            best_match_path = None
            best_match_count = 0

            # 폴더에서 book0.jpg ~ book72.jpg 모두 불러오기
            img_paths = sorted(glob.glob("../img/book*.jpg"))

            for path in img_paths:
                db_img = cv2.imread(path)       # 비교할 데이터 이미지 읽기
                kp2, des2 = orb.detectAndCompute(db_img, None)  # 특징점 추출
                if des1 is None or des2 is None:
                    continue                   # 특징점 없으면 건너뛰기

                # 두 이미지 디스크립터 매칭
                matches = bf.match(des1, des2)

                # 거리가 가까운 순으로 정렬
                matches = sorted(matches, key=lambda x: x.distance)

                # 일정 거리 이하인 좋은 매칭 필터링 (50 이하)
                good_matches = [m for m in matches if m.distance < 50]

                # 좋은 매칭 개수가 가장 많은 이미지 저장
                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                    best_match_path = path

            # 매칭 가장 잘된 이미지 보여주기
            if best_match_path:
                best_img = cv2.imread(best_match_path)
                cv2.imshow("Best Match", best_img)
                print(f"Best match: {best_match_path} with {best_match_count} good matches")

cap.release()
cv2.destroyAllWindows()
