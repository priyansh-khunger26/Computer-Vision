import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.read('face_trained.yml')

# Specify the path here
img = cv.imread('/Users/priyansh/Desktop/Downloads/opencv-course-master/Resources/Faces/val/madonna/1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Now again detect the face in the gray scale image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recogniser.predict(faces_roi)
    print(f"Label = {people[label]} with a confidence of {confidence}")

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv.imshow('Detected Face', img)

cv.waitKey(0)

