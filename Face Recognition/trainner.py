import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
path = 'dataset'
print (cv2.__version__)
def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        # openCV only works with numpy
        faceImg = Image.open(imagePath).convert('L')  # converted to into grayScale, this now in PIL
        # converted into numpy array
        faceNp = np.array(faceImg, 'uint8')

        # get id from dataSet
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        print(ID)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

IDs,faces = getImageWithID(path)
recognizer.train(faces, IDs)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
