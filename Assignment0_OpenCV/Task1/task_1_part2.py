import cv2 as cv
import numpy as np
import os
import natsort 

image_directory = "/Users/priyansh/Desktop/cv assignment 0/task_1/images created"
video_directory = "/Users/priyansh/Desktop/cv assignment 0/task_1/video created"

# final_video = cv.VideooWriter("final_video.mp4", cv.VideoWriter_fourcc(*'XVID'), )

lists = []

for i in os.listdir(image_directory):
    lists.append(i)

print(lists)

new_list = natsort.natsorted(lists)
print(new_list)

img_array = []
for image in new_list:
    img_path = os.path.join(image_directory, image)
    img = cv.imread(img_path)
    img_array.append(img)
    print(image)

print(len(img_array))


height, width, layers = img_array[1].shape
fps = 28.92
os.chdir(video_directory)
final_video = cv.VideoWriter("Final_video.mp4", cv.VideoWriter_fourcc(*'avc1'), fps, (width, height))

for i in range(len(img_array)):
    if i == 0:
        continue
    final_video.write(img_array[i])

final_video.release()
cv.destroyAllWindows()
