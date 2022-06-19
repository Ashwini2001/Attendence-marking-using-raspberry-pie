'''Face Detection using HaarCascade'''

from  imutils.video import VideoStream
import datetime
import imutils
import time
import cv2
import requests
import os
from datetime import date
import csv
#import RPi.GPIO as gpio

# Start video

vs = VideoStream(0).start()
time.sleep(2.0)
# gpio.setwarnings(False)
# gpio.setmode(gpio.BOARD)
today = date.today()
d = today.strftime("%d/%m/%Y")
# Font for text on image
font = cv2.FONT_HERSHEY_SIMPLEX

# Load Haar Classifiers for face and mouth detection
face_cascade = cv2.CascadeClassifier('../Cascade_files/haarcascade_frontalface_default.xml')

# Input ID
ID = raw_input("Enter your Roll number: ")

# Input name
name = raw_input("Enter your name: ")

year = int(input("Enter year of study: "))

Branch = raw_input("Enter Branch: ")

line = [d, name, ID, year, Branch]
try:
    os.mkdir("../Dataset/" + "User" + ID)
except Exception as e:
    pass

sampleNum = 0
while True:
	# Read frame by frame
    frame = vs.read()

    # Show the frame
    cv2.imshow("Feed",frame)

    # Covert to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define waitKey
    key = cv2.waitKey(1) & 0xFF

    # Detect faces using classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

    # If face not found
    if(len(faces) == 0):

        print("Face not detected.Please try again!")
        continue
    else:

    	# For every face detected
        for (x, y, w, h) in faces:

        	# Draw a rectangle over the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)

            sampleNum = sampleNum + 1

            # Save the detected face
            cv2.imwrite("../Dataset/User" + ID + "/" + str(ID) + "." + str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
            
    # Press q to exit the loop
    if key == ord("q"):
        break
    elif sampleNum>=20:
        break

# Stop video
vs.stop()

# Destroy the windows created
cv2.destroyAllWindows()

with open('../CSV_files/Student_data.csv', 'a+') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(line)
    writeFile.close()
