##### Hello, I'm Nguyen Nhut Tin.

'''----------------------------------------------------------------------------------------------'''
##### Lib
# System
import os
import numpy as np
import cv2
import pandas as pd
from sklearn import preprocessing
import sys
import time
import warnings

# CNN
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# Data preprocessing 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
##### END-LIB
'''----------------------------------------------------------------------------------------------'''




'''----------------------------------------------------------------------------------------------'''
##### Preparing the data
# Generate the data
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)


# Read the data from folders 
human_faces = []
answers = []

location = './images' # Root folder
directory = os.listdir(location)
directory = sorted(directory) # Arrange the data
for i in directory:
	path = location + '/' + i
	sub_dir = os.listdir(path)
	for j in sub_dir:
		# Add the first picture and label into arrays
		image = cv2.imread((path+'/'+j))
		human_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		human_face = cv2.resize(human_face, dsize=(128, 128))
		human_faces.append(human_face)
		answers.append(i)

		# Run
		human_face = np.expand_dims(image, axis=0)
		aug_iter = datagen.flow(human_face)
		aug_images_arr = [next(aug_iter)[0].astype(np.uint8) for i_aug in range(11)] # Pull the data out of box
		for insi in aug_images_arr:
			human_face = cv2.cvtColor(insi, cv2.COLOR_BGR2GRAY)
			human_face = cv2.resize(human_face, dsize=(128, 128))
			human_faces.append(human_face)
			answers.append(i)
		

# convert data from unit to float32
human_faces = np.array(human_faces).astype('float32')

# dims=4 ()
human_faces= np.expand_dims(human_faces, axis=4)

# LabelEncoder (from character to number)
le = preprocessing.LabelEncoder()
answers = le.fit_transform(answers)
answers = np.array(answers)

# making Random() data
human_faces, answers = shuffle(human_faces, answers)

# Split the dataset
X_train_raw, X_final_test, y_train_raw, y_final_test = train_test_split(human_faces, answers, test_size=0.15)

# Reduce the values , easy to work with
X_final_test = X_final_test / 255.0

# Convert vector(int) to binary matrix
y_final_test = np_utils.to_categorical(y_final_test)
'''----------------------------------------------------------------------------------------------'''

##### Get the quantity of objects
num_classes = y_final_test.shape[1]



'''----------------------------------------------------------------------------------------------'''
##### CNN architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128,128,1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))


'''----------------------------------------------------------------------------------------------'''
##### Compile model
epochs = 100
lrate = 0.01
decay = lrate/epochs 
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


'''----------------------------------------------------------------------------------------------'''
##### Fit the real data into model
k = 5
folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train_raw, y_train_raw))
X_train_raw = X_train_raw / 255.0
y_train_raw = np_utils.to_categorical(y_train_raw)

for j, (train_idx, val_idx) in enumerate(folds):
	print('\nFold ',j)
	X_train = X_train_raw[train_idx]
	y_train = y_train_raw[train_idx]
	X_val   = X_train_raw[val_idx]
	y_val   = y_train_raw[val_idx]

	##########
	model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=128)
	##########


'''----------------------------------------------------------------------------------------------'''
##### Accuracy of model
result = model.evaluate(X_final_test, y_final_test, verbose=0)
print("Test_Acc: %.2f%%" % (result[1]*100))
'''----------------------------------------------------------------------------------------------'''

##### To json file
cv_to_json = model.to_json()
with open("./output/model.json", "w") as file:
    file.write(cv_to_json)
##### To HDF5 file
model.save_weights("./output/model.h5")
print("Saved model to disk")
'''----------------------------------------------------------------------------------------------'''
#################### END ####################