##### Hello, I'm Nguyen Nhut Tin.

'''----------------------------------------------------------------------------------------------'''
##### Lib
# System
import requests
import json
import cv2
import jsonpickle
import warnings
from pathlib import Path
from PIL import Image

# Interface
from tkinter import *        
from PIL import ImageTk, Image
import easygui
from tkinter import messagebox
##### END-LIB
'''----------------------------------------------------------------------------------------------'''



'''----------------------------------------------------------------------------------------------'''
##### Draw on images
# Draw boxes
def draw_rectangle(img, x,y,w,h):
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Draw labels
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
'''----------------------------------------------------------------------------------------------'''




'''----------------------------------------------------------------------------------------------'''
##### Tkinter
def toFileFunc(data):
	name = easygui.enterbox("File name (.txt): ")
	if name:
		file = Path("text/"+name+".txt")
		if file.is_file():
			messagebox.showerror("Error", "File is already exist")
		else:	
			mfile = open(file,"w") 
			for d in data['results']:
				mfile.write(d['label'] +'\n')

			mfile.close() 
			messagebox.showinfo("Information","File Successfully Saved")




def run_interface(frame_f,data):
	'''----------------------------------------------------------------------------------------------'''
	########## Root frame ##########
	root = Tk()
	########## Root frame ##########
	'''----------------------------------------------------------------------------------------------'''


	'''----------------------------------------------------------------------------------------------'''
	##### Image Frame
	img = Text(root, height=30, width=80)
	# Convert Cv2 to PIL image
	photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB)))
	img.image_create(END, image=photo)
	img.pack(side=LEFT)


	##### Second Frame #####
	frame = Frame(root)
	frame.pack()
	##### Second Frame #####


	# TextField Areaa #
	textField = Text(frame, height=25, width=50)
	scroll = Scrollbar(frame, command=textField.yview)
	textField.configure(yscrollcommand=scroll.set)
	# TextField Areaa #


	# Insert
	for d in data['results']:
		textField.insert(END,'\n '+ d['label'] +'\n')



	# UP #
	textField.pack(side=LEFT)
	# UP #


	## DOWN ##
	# Small Frame #
	bottomframe = Frame(root)
	bottomframe.pack(side = BOTTOM)
	# Small Frame #

	# Add button to Small Frame #
	toFile = Button(bottomframe, text="ToFile",command=lambda d=data: toFileFunc(d))
	toFile.pack(side=RIGHT, pady=25,padx=5)
	## DOWN ##

	# Add scrollBar #
	scroll.pack(side=RIGHT, fill=Y)
	# Add scrollBar #
	'''----------------------------------------------------------------------------------------------'''

	########## Main ##########
	root.mainloop()
	########## Main ##########
	
'''----------------------------------------------------------------------------------------------'''




def main():

	'''----------------------------------------------------------------------------------------------'''
	##### Client windows
	video_capture = cv2.VideoCapture(0)
	while True:
		# Capture frame-by-frame
		ret, frame = video_capture.read()
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		try:
			url = 'http://localhost:5000/'

			# prepare headers for http request
			headers = {'content-type': 'image/jpeg'}

			# encode image as jpeg
			_, img_encoded = cv2.imencode('.jpg', frame)


			########## send http request with image and receive response ##########
			response = requests.post(url, data=img_encoded.tostring(), headers=headers)
			########## send http request with image and receive response ##########


			# decode response
			data = json.loads(response.text)
			
			# Receive the data from server.
			for d in data['results']:
				x = int(d['x'])
				y = int(d['y'])
				w = int(d['w'])
				h = int(d['h'])
				draw_rectangle(frame, x,y,w,h)
				draw_text(frame, d['label'], x, (y-5))

		except:
			continue


		# Display the resulting frame
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	# When everything is done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()


	########### Result ###########
	run_interface(frame, data)
	########### Result ###########

	'''----------------------------------------------------------------------------------------------'''
	##### END

if __name__ == '__main__':
    main()  