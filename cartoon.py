import numpy as np
import cv2
from matplotlib import pyplot as plt 

img = plt.imread('D:\Machine learning\cartoon Effect/image.jpg')
#this will show image in bgr
plt.imshow(img)
plt.show()


#so we will convert it in rgb
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

#as we can use bilateral Fiter but it is computional expencive so we reduce the image of pixel
img_samll = cv2.pyrDown(image)
#pyrDown is used to reduce the image pixel.In this we will remove the even number rows and even number colom which is known as primade reduce image

num_iter = 5
for _ in range(num_iter):
    imag_small = cv2.bilateralFilter(img_samll, d=9, sigmaColor=9, sigmaSpace=7)
#d is kernal which is 3x3 5x5 anything 
#sigmaColor is for pixel sigma
#sigmaSpace is for positional value
img_rgb = cv2.pyrUp(img_samll)
plt.imshow(img_rgb)
plt.show()

#edge line
#converting the colour of image from rgb to gray
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#using median blur so edges can be highlited 
img_blur = cv2.medianBlur(img_gray,7)
#7 is kernal size

img_edge = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,2)
#2 is gaussion average 7 is kernal
plt.imshow(img_edge)
plt.show()

img_edge = cv2.cvtColor(img_edge,cv2.COLOR_GRAY2RGB)
plt.imshow(img_edge)
plt.show()

array = cv2.bitwise_and(image, img_edge)
plt.figure(figsize= (10,10))
plt.imshow(array)

plt.show()