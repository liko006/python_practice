
import matplotlib.pyplot as plt
import cv2
from mosaic import mosaic as mosaic

# 캐스케이드 파일을 지정해서 검출기 만들기
cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

# 이미지를 읽어들인 후 그레이스케일로 변환하기
img = cv2.imread('girl2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 인식하기
face_list = cascade.detectMultiScale(img_gray, minSize=(150,150))

# 결과 확인하기
if len(face_list) == 0 :
    print("얼굴 인식 실패")
    quit()

# 인식성공시 인식한 부분 출력하기
for (x, y, w, h) in face_list :
    img = mosaic(img, (x,y,x+w,y+h), 10)
    
# 이미지 출력
cv2.imwrite('face-detect.png', img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
