import cv2
import numpy as np
import glob

img1 = None  # ROI 이미지
win_name = 'Camera Matching'
MIN_MATCH = 10

# ORB 검출기 + FLANN 매처
detector = cv2.ORB_create(1000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 전처리 함수: 해시 계산 전에 선명도 증가 등 처리
def preprocess_for_hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)  # 선명도 증가
    return sharp

# 평균 해시 함수 (전처리 적용)
def img2hash(img):
    proc = preprocess_for_hash(img)
    resized = cv2.resize(proc, (16, 16))
    avg = resized.mean()
    bi = 1 * (resized > avg)
    return bi

# 해밍 거리 함수
def hamming_distance(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    return (a != b).sum()

# 카메라 연결
cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():       
    ret, frame = cap.read()
    if not ret:
        break

    if img1 is None:
        res = frame
    else:
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            res = frame
        else:
            matches = matcher.knnMatch(desc1, desc2, 2)
            ratio = 0.75
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < n.distance * ratio:
                        good_matches.append(m)
            print('good matches:%d/%d' % (len(good_matches), len(matches)))

            matchesMask = None
            if len(good_matches) > MIN_MATCH:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mtrx is not None:
                    accuracy = float(mask.sum()) / mask.size
                    print("accuracy: %d/%d(%.2f%%)" % (mask.sum(), mask.size, accuracy * 100))
                    if mask.sum() > MIN_MATCH:
                        matchesMask = [int(x) for x in mask.ravel()]
                        cv2.putText(img2, "matching success!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        h, w = img1.shape[:2]
                        pts = np.float32([[[0, 0]], [[0, h-1]], [[w-1, h-1]], [[w-1, 0]]])
                        dst = cv2.perspectiveTransform(pts, mtrx)
                        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(img2, "matching failure", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                  matchColor=(0, 255, 0), matchesMask=matchesMask,
                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 종료
        break
    elif key == ord(' '):
        # ROI 선택
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]

            # 해시 비교 시작
            query_hash = img2hash(img1)
            img_paths = sorted(glob.glob("../img/book*.jpg"))
            best_path = None
            min_dist = 9999

            for path in img_paths:
                db_img = cv2.imread(path)
                if db_img is None:
                    continue
                db_hash = img2hash(db_img)
                dist = hamming_distance(query_hash, db_hash)
                print(f"{path}: 해밍 거리 = {dist}")
                if dist < min_dist:
                    min_dist = dist
                    best_path = path

            if best_path:
                best_img = cv2.imread(best_path)
                cv2.imshow("Best Match", best_img)
                print(f"success: {best_path} (해밍 거리: {min_dist})")

cap.release()
cv2.destroyAllWindows()
