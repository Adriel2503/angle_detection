import numpy as np
import cv2

imagen=cv2.imread('../data/sensor.jpg')
imagen=cv2.resize(imagen,None,fx=0.6,fy=0.5,interpolation=cv2.INTER_CUBIC)

#Convertimos la imagen de RGB a HSV
hsv=cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)

#Definimos el intervalo de tonalidad
lower_green=np.array([0, 0, 0])
upper_green=np.array([20, 255, 255])

#Detecta los tonos HSV en la imagen
mask=cv2.inRange(hsv,lower_green,upper_green)
kernel=np.ones((3,3),np.uint8)
erosion=cv2.erode(mask,kernel,iterations=2)
#dilation=cv2.dilate(mask,kernel,iterations=1)

#Mostramos la imagen real extraida
tierra=cv2.bitwise_and(imagen,imagen,mask=mask)

contours,_=cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    area=cv2.contourArea(c)
    if area>500:
        cv2.drawContours(imagen,contours,-1,(0,0,255),cv2.LINE_4)

cv2.imshow('img',imagen)
cv2.waitKey(0)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
#cv2.imshow('dilatacion',dilation)
#cv2.waitKey(0)
cv2.imshow('objeto',tierra)
cv2.waitKey(0)

cv2.destroyAllWindows()