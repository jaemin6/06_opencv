import cv2
import numpy as np
import glob

# ORB 생성기
orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord(' '):
        x, y, w, h = cv2.selectROI("Camera", frame, False)
        if w and h:
            roi = frame[y:y+h, x:x+w]
            kp1, des1 = orb.detectAndCompute(roi, None)

            best_match_path = None
            best_match_count = 0

            img_paths = sorted(glob.glob("../img/book*.jpg"))
            for path in img_paths:
                db_img = cv2.imread(path)
                kp2, des2 = orb.detectAndCompute(db_img, None)
                if des1 is None or des2 is None:
                    continue

                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # 좋은 매칭 개수 기준
                good_matches = [m for m in matches if m.distance < 50]

                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                    best_match_path = path

            if best_match_path:
                best_img = cv2.imread(best_match_path)
                cv2.imshow("Best Match", best_img)
                print(f"Best match: {best_match_path} with {best_match_count} good matches")

cap.release()
cv2.destroyAllWindows()
