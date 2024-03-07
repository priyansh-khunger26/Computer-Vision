import cv2 as cv
import numpy as np

img = cv.imread('/Users/priyansh/Downloads/WhatsApp Image 2023-12-03 at 23.10.20.jpeg')
cv.imshow('Abhinav', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 2)
print(f"Number of faces in the image are: {len(faces_rect)}")
print(faces_rect.shape)

for (a,b,c,d) in faces_rect:
    cv.rectangle(img, (a,b), (a+c, b+d), (0,255,0),thickness=2)

cv.imshow('face detection', img)

cv.waitKey(0)