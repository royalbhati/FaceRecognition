import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random


facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
















# img=cv2.imread('image.jpg',0)#to load any image 0 -indicates Grayscale

# cv2.imshow('image',img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()

# plt.imshow(img,cmap='gray')
# plt.plot([50,100],[90,100],[100,200],'r',linewidth=10)
# plt.show()
# cv2.imwrite('gooo.jpg',img)

# fourcc=cv2.VideoWriter_fourcc(*'XVID')#codec for video
# out=cv2.VideoWriter('out.avi',fourcc,20.0,(640,480))#output attributes
id=random.randint(0,100)
sample=0
while True:


	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	face=facedetect.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in face:
		sample+=1
		cv2.imwrite('dataset/'+str(id)+'.'+str(sample)+'.jpg',gray[y:y+h,x:x+w])
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(100 )

	# out.write(frame)
	cv2.imshow('frame',frame)
	


	if sample>20:
		break

cap.release()
# out.release()		
cv2.destroyAllWindows()
