"""
1. 캠이 열리면 스페이스바로 사진을 촬영
2. 사용자가 ROI(관심영역) 선택
3. 선택한 이미지(ROI)에서 특징점 추출
4. 폴더 내 book0.jpg ~ book72.jpg 이미지들 중에서 매칭 가장 좋은 이미지 찾기
5. 매칭이 가장 좋은 이미지를 화면에 보여줌
6. esc 누르면 종료
"""

import cv2
import numpy as np
import glob

# ORB 특징점 검출기 생성 (1000개 특징점까지)
orb = cv2.ORB_create(1000)  # ORB 특징점 및 디스크립터 추출기 생성

# BFMatcher 생성 (해밍 거리 사용, 교차 검증=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # BFMatcher 생성, 해밍거리 기준

# 카메라 열기 (0번 장치)
cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 카메라 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 카메라 프레임 높이 설정

while True:
    ret, frame = cap.read()        # 카메라에서 한 프레임 읽기
    if not ret:
        break                     # 프레임 읽기 실패 시 종료

    cv2.imshow("Camera", frame)   # 원본 카메라 화면 보여주기
    key = cv2.waitKey(1) & 0xFF  # 키 입력 대기 및 입력 받기

    if key == 27:  # ESC 키 누르면 종료
        break

    elif key == ord(' '):  # 스페이스 바 눌렀을 때
        # 사용자에게 ROI(관심영역) 선택 요청 (마우스로 드래그하여 선택)
        x, y, w, h = cv2.selectROI("Camera", frame, False)
        if w and h:  # 선택된 영역이 있을 경우
            roi = frame[y:y+h, x:x+w]  # 선택된 영역 이미지 자르기

            # ROI 이미지에서 ORB 특징점과 디스크립터 계산
            kp1, des1 = orb.detectAndCompute(roi, None)

            best_match_path = None   # 가장 잘 맞는 이미지 경로 저장 변수 초기화
            best_match_count = 0     # 가장 좋은 매칭 개수 초기화

            # 폴더에서 book0.jpg ~ book72.jpg 모두 불러오기 (파일명 순서대로)
            img_paths = sorted(glob.glob("../img/book*.jpg"))

            for path in img_paths:
                db_img = cv2.imread(path)            # 데이터베이스 이미지 읽기
                kp2, des2 = orb.detectAndCompute(db_img, None)  # 데이터베이스 이미지 특징점 추출
                if des1 is None or des2 is None:
                    continue                        # 특징점 없으면 건너뛰기

                # 두 이미지 디스크립터 매칭
                matches = bf.match(des1, des2)

                # 매칭 결과를 거리순으로 정렬 (거리가 가까울수록 좋은 매칭)
                matches = sorted(matches, key=lambda x: x.distance)

                # 거리 50 이하인 좋은 매칭만 필터링
                good_matches = [m for m in matches if m.distance < 50]

                # 좋은 매칭 개수가 가장 많으면 갱신
                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                    best_match_path = path

            # 가장 잘 매칭된 이미지가 있으면
            if best_match_path:
                best_img = cv2.imread(best_match_path)  # 이미지 읽기
                cv2.imshow("Best Match", best_img)      # 이미지 창으로 보여주기
                print(f"Best match: {best_match_path} with {best_match_count} good matches")

# 카메라 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
