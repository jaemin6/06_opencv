import cv2
import numpy as np
import glob

# 전처리 함수 (선명도 올리기)
def preprocess_for_hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharp

# 평균 해시 계산
def img2hash(img):
    proc = preprocess_for_hash(img)
    resized = cv2.resize(proc, (16,16))
    avg = resized.mean()
    return 1 * (resized > avg)

# 해밍 거리 계산
def hamming_distance(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    return (a != b).sum()

# 캠 연결
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
    elif key == ord(' '):  # 스페이스바: ROI 선택 & 매칭 시작
        x, y, w, h = cv2.selectROI("Camera", frame, False)
        if w and h:
            roi_img = frame[y:y+h, x:x+w]
            
            # ROI 이미지 해시 계산
            roi_hash = img2hash(roi_img)
            
            # book 이미지 경로 리스트
            img_paths = sorted(glob.glob("../img/book*.jpg"))
            
            best_path = None
            min_dist = 9999
            
            for path in img_paths:
                db_img = cv2.imread(path)
                if db_img is None:
                    continue
                db_hash = img2hash(db_img)
                dist = hamming_distance(roi_hash, db_hash)
                
                if dist < min_dist:
                    min_dist = dist
                    best_path = path
            
            if best_path:
                best_img = cv2.imread(best_path)
                cv2.imshow("Best Match", best_img)
                print(f"Best Match: {best_path} (Hamming distance: {min_dist})")

cap.release()
cv2.destroyAllWindows()
