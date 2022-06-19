import os
import cv2
import numpy as numpy


def recognize(path):
	recognizer = cv2.face.createLBPHFaceRecognizer()
	# Load Recognizer
	recognizer.load("../Trainner/" + path + "/trainner.yml")

	cascadepath = '../Cascade_files/haarcascade_frontalface_default.xml'
	faceCascade = cv2.CascadeClassifier(cascadepath)

	cam = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	while True:
	    ret, im =cam.read()
	    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	    faces=faceCascade.detectMultiScale(gray, 1.2,5)
	    for(x,y,w,h) in faces:
	        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
	        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
	        if(conf>40):
	            Id = str(folder)
	        else:
	            Id="Unknown"
	        cv2.putText(im,str(Id), (x,y+h),font,  0.5, (155, 120, 255))
	    cv2.imshow('im',im) 
	    if cv2.waitKey(10) & 0xFF==ord('q'):
	        break
	cam.release()
	cv2.destroyAllWindows()

Id = raw_input("Enter your Roll number: ")
name = raw_input("Enter your name: ")
folders = os.listdir("../Trainner")
for folder in folders:
	# print(Id,folder)
	if(("User" + str(Id))==str(folder)):
		recognize(folder)
print("No data found")
field_names=['Time stamp', 'Name', 'PRN', ]
with open('../CSV_files/Attendance_sheet.csv', 'a+') as writeFile:
    writer = csv.writer(writeFile,fieldnames=field_names)
    writer.writerow(line)
    writeFile.close()