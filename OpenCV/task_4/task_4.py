import cv2 as cv
import numpy as np

vid = cv.VideoCapture("/Users/priyansh/Downloads/WhatsApp Video 2024-01-09 at 15.59.49.mp4")

haar_cascade = cv.CascadeClassifier('/Users/priyansh/Desktop/haar_face.xml')

while True:
    ret, frame = vid.read()

    if ret == False:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.3, 5)
    for (a,b,c,d) in faces_rect:
        cv.rectangle(frame, (a,b), (a+c, b+d), (0,255,0),thickness=2)
    
    cv.imshow('face detection', frame)
    # cv.waitKey(0)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows