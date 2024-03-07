import cv2 as cv
import numpy as np
import os

cap = cv.VideoCapture('/Users/priyansh/Documents/cv assignment 1/WhatsApp Video 2024-01-09 at 15.59.49.mp4')

i = 0
while(cap.isOpened()):
  ret, frame = cap.read()

  if ret == True:
    cv.imwrite('/Users/priyansh/Desktop/cv assignment 0/task_1/images created/frame' + str(i) + '.jpg', frame)
    cv.imshow('Frame',frame)
 
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break
  i = i + 1

cap.release()
cv.destroyAllWindows()