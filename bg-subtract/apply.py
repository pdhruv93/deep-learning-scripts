import cv2
import numpy as np

IMG_PATH = ""

image = cv2.imread('C://deep learning//fingers-count-scratch//to_predict_images//1L.png')

#change color space from BGR-->HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# perform Thresholding--Create a binary image where white will be skin colors and rest is black
#check: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
#you can say that for filtering skin color we have color range [2, 0, 0] to [20, 255, 255]--atleast I could not figure it out
image = cv2.inRange(hsv, np.array([-30, -30, 100]), np.array([50, 50, 250]))


cv2.imshow('Masked Image', image)
cv2.waitKey(0)


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
