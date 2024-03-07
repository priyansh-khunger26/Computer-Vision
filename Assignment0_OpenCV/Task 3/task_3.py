import cv2 as cv
import numpy as np 

video1 = cv.VideoCapture("mixkit-weather-presenter-giving-newsroom-presentation-on-a-green-screen-28292-medium.mp4") 
video2 = cv.VideoCapture("sample-10s.mp4") 

while True: 
	ret1, frame1 = video1.read() 
	ret2, frame2 = video2.read() 
	
	if ret1 == False or ret2 == False:
		break
	hsv = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
	hsv = cv.resize(hsv, (640, 480)) 
	frame1 = cv.resize(frame1, (640, 480))
	frame2 = cv.resize(frame2, (640, 480)) 

	u_green = np.array([80, 255, 255]) 
	l_green = np.array([40, 40, 40]) 

	mask = cv.inRange(hsv, l_green, u_green) 
	res = cv.bitwise_and(frame1, frame1, mask = mask) 

	f = frame1 - res 
	f = np.where(f == 0, frame2, f) 

	cv.imshow("video", frame1) 
	cv.imshow("mask", f) 

	if cv.waitKey(25) == 27: 
		break

video1.release() 
video2.release()
cv.destroyAllWindows() 
