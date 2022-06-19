import cv2,os
import numpy as np
from PIL import Image


recognizer = cv2.face.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier("../Cascade_files/haarcascade_frontalface_default.xml")

path = "../Dataset"


def getImagesAndLabels(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	faceSamples = []
	Ids = []

	for imagePath in imagePaths:

		# Loading the image and converting it to gray scale
		pilImage = Image.open(imagePath).convert('L')

		# Convert PIL image to numpy array
		imageNp = np.array(pilImage, 'uint8')

		# get Id from the image
		Id = int(os.path.split(imagePath)[-1].split(".")[1])

		# Extract face from training image sample
		faces = detector.detectMultiScale(imageNp)

		# If a face is there, then append in the list as well as Id
		for (x,y,w,h) in faces:
			faceSamples.append(imageNp[y:y+h,x:x+w])
			Ids.append(Id)

	return faceSamples,Ids

# Determining subfolders in Dataset
imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

# Iterate over the students folder containing dataset
for imagePath in imagePaths:

	faces,Ids = getImagesAndLabels(imagePath)

	# train model
	recognizer.train(faces,np.array(Ids))

	# Student's folder name
	folder_name = imagePath.split('/')[-1]
	try:
		# make directory of each student in trainner folder
		os.mkdir("../Trainner/" + folder_name)
	except Exception as e:
		pass

	# Save yml file
	recognizer.save('../Trainner/' + folder_name +'/trainner.yml')