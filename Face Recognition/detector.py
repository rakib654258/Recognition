import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./recognizer/trainingData.yml')
id = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.5, 3)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 100), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if id==1:
            id = 'Rakib'
        elif id ==2:
            id = 'Jony'
        else:
            id = 'Unknown'
        cv2.putText(img, str(id), (x,y+h), font,3,(100,150,220),2)


    cv2.imshow('Faces', img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()