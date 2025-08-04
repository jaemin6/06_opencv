import cv2
import numpy as np
import glob

# 이미지 해시 함수 정의
def img2hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    avg = gray.mean()
    return 1 * (gray > avg)

# 해밍 거리 계산 함수
def hamming_distance(a, b):
    return (a.reshape(-1) != b.reshape(-1)).sum()

# 비교 대상 이미지 로딩 및 해시 계산
search_dir = '../img/books'
img_paths = sorted(glob.glob(search_dir + '/book*.jpg'))
hashes = []
images = []

for path in img_paths:
    img = cv2.imread(path)
    h = img2hash(img)
    hashes.append(h)
    images.append(img)

# 카메라로 한 장 캡처
cap = cv2.VideoCapture(0)
cv2.namedWindow('Camera')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1)

    if key == 32:  # 스페이스바 눌러서 촬영
        query = frame.copy()
        break
    elif key == 27:  # ESC
        query = None
        break

cap.release()
cv2.destroyAllWindows()

if query is not None:
    query_hash = img2hash(query)

    min_dist = 1e9
    match_img = None

    for h, img in zip(hashes, images):
        dist = hamming_distance(query_hash, h)
        if dist < min_dist:
            min_dist = dist
            match_img = img

    if match_img is not None:
        cv2.imshow('Best Match', match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
