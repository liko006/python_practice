# 파일명 : mosaic.py
# 모자이크 하는 함수를 모듈로 저장

import cv2

def mosaic(img, rect, size):
    
    # 모자이크 적용할 부분 추출하기
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]
    
    # 축소후 확대
    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w,h), interpolation=cv2.INTER_AREA)
    
    # 모자이크 적용
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    
    return img2
