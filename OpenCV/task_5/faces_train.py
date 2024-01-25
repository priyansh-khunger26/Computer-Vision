import os
from pyexpat import features
import cv2 as cv
import numpy as np

DIR = r"/Users/priyansh/Desktop/Downloads/opencv-course-master/Resources/Faces/train/"

people = []
for i in os.listdir(DIR):
    people.append(i)

print(people)

features_faces = []
labels = []

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for person in people:
        label = people.index(person)
        path = os.path.join(DIR, person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features_faces.append(faces_roi)
                labels.append(label)


create_train()
print("Training done-----------------------")

# Till here we have only created the data-set
print(f"Lenght of the features list: {len(features_faces)}")
print(f"Length of the labels list: {len(labels)}")

# Now we will be doing the training part using OpenCV
features = np.array(features_faces, dtype="object")
labels = np.array(labels)

face_recogniser = cv.face.LBPHFaceRecognizer_create()
face_recogniser.train(features_faces, labels)
face_recogniser.save('face_trained.yml')

# np.save('features.npy', features)
# np.save('labels.npy', labels)
