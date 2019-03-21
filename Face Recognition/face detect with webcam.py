import cv2
import numpy as np

# Classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while(True):
# gray image
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.5, 3)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 100), 2)
        cv2.imshow("Face", img)
        cv2.imshow('Gray Face', gray)
    if (cv2.waitKey(1)==ord('g')):
        break
cam.release()
cv2.destroyAllWindows()
