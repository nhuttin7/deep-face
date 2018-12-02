##### Hello, I'm Nguyen Nhut Tin.

'''----------------------------------------------------------------------------------------------'''
##### Lib
# System and Computer Vision platform
from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import time
import cv2
import os
import numpy as np
import sys
import warnings
import json
import tensorflow as tf

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# Flask framework
from flask import Flask, request, Response

# Json tool
import jsonpickle

##### END-LIB
'''----------------------------------------------------------------------------------------------'''





'''----------------------------------------------------------------------------------------------'''
# Initialize the Flask application
app = Flask(__name__)

# load json and model
json_file = open('./output/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
global loaded_model 
loaded_model = model_from_json(loaded_model_json)
global graph
graph = tf.get_default_graph()

# load weights into new model
loaded_model.load_weights("./output/model.h5")
print("Loaded model from disk")

'''----------------------------------------------------------------------------------------------'''




'''----------------------------------------------------------------------------------------------'''
##### route http posts to this method
@app.route('/', methods=['POST'])
def index():
	try:
		# Get labels
		location = 'images'
		directory = os.listdir(location)

		# convert string of image data to uint8
		nparr = np.fromstring(request.data, np.uint8)

		# decode image
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

		########## CBC DETECTION ##########
		picture, human_faces = reconigze_face(img) 
		########## CBC DETECTION ##########

		size = len(human_faces) 
		data = {}  
		data['results'] = []  
		if(human_faces is not None):
			for j in range(0,size):
				i = human_faces[j]
				(x,y,w,h) = i
				faces = np.array(picture)
				faces = imutils.resize(picture[y:y+w, x:x+h], width=128)
				faces = faces.astype('float32')
				faces /= 255.0
				faces= np.expand_dims([faces], axis=4)

				########## CNN PREDICTION ##########
				with graph.as_default():
					answer = loaded_model.predict(faces)
				########## CNN PREDICTION ##########
				
				########## CONDITION RETURN ##########
				if np.amax(answer) > 0.999 :
					# Create json file
					(x,y,w,h) = i
					data['results'].append({  
						'label': directory[np.argmax(answer)],
						'x': str(x), 
						'y': str(y),
						'w': str(w),
						'h': str(h) 
					})
				else:
					# Create json file
					(x,y,w,h) = i
					data['results'].append({  
						'label': str(""),
						'x': str(x), 
						'y': str(y),
						'w': str(w),
						'h': str(h) 
					})



		# Encode response using jsonpickle
		response = jsonpickle.encode(data)
		# Back to Client
		return Response(response=response, status=200, mimetype="application/json")

	except:
		print ("NULL")

    
'''----------------------------------------------------------------------------------------------'''




'''----------------------------------------------------------------------------------------------'''
##### CBC detection function 
def reconigze_face(image):
    picture = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('./lib/haarcascade_frontalface_alt.xml')
    human_faces = cascade.detectMultiScale(picture, scaleFactor=1.1,minNeighbors=5)
    if(len(human_faces)==0):
        return None, None
    return picture, human_faces
'''----------------------------------------------------------------------------------------------'''




'''----------------------------------------------------------------------------------------------'''
##### Main function
if __name__ == '__main__':
    app.run()
'''----------------------------------------------------------------------------------------------'''