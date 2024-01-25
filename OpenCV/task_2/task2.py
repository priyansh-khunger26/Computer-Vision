import cv2 as cv
import numpy as np

vid = cv.VideoCapture(0)

i = 0

while vid.isOpened():
    ret, frame = vid.read()

    if ret == False:
        break

    cv.imshow(f"frame{i}", frame)

    cv.imwrite(f"/Users/priyansh/Desktop/cv assignment 0/task_2/images created/frame{i}.jpg", frame)
    i = i + 1

    if cv.waitKey(25) & 0xFF == ord('q'):
        break


vid.release()
cv.destroyAllWindows()