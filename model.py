import os
import csv

# read csv file
samples = []
with open('/Users/jiawenzhu/Desktop/self-driving/Project4/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    
# prepare training and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import math
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '/Users/jiawenzhu/Desktop/self-driving/Project4/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                
                # convert to RGB
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                flipped_center_image = np.fliplr(center_image)
                images.append(flipped_center_image)
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# set batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
print('Data Loaded')

# build a more powerful network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
ch, row, col = 160, 320, 3 # trimmed image format

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,
        input_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=2, verbose=1)

model.save('model.h5')
