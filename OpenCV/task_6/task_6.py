import os
from pickle import TRUE
import numpy as np
import cv2 as cv
import gc
import caer
import canaro
from tensorflow.keras.utils import to_categorical

IMG_SIZE = (80, 80)
channels = 1
char_path = "/Users/priyansh/Desktop/Downloads/archive/simpsons_dataset"

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

char_dict = caer.sort_dict(char_dict, descending = True)
# print(char_dict)

# Now we are gonna create the data set
characters = []
count = 0
for i in char_dict:
    characters.append(i[0]) 
    count = count + 1
    if count >= 10:
        break

# print(characters)

train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)
print(len(train))

featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)
print(len(featureSet), len(labels))

featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))
