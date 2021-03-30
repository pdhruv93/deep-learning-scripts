from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img


#Global Declarations
MODEL_PATH = "finger_count_model.h5"
IMAGE_PATH = "./to_predict_images/2L.png"


#Load Image
image = cv2.imread(IMAGE_PATH)
copy = image.copy()
copy = cv2.resize(copy, (128, 128)) #this is just done bcoz the image was too big in the final output window and there was no minimize option

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (128, 128))
image = cv2.GaussianBlur(image, (7,7), 3)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
ret, image = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#image = cv2.inRange(hsv, np.array([-10, 50, 100]), np.array([50, 150, 225]))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("Output", image)
cv2.waitKey(0)
image = img_to_array(image)
image = image * 1./255
image = np.expand_dims(image, axis=0)

print(image.shape)



#Load model and Predict
class_names = ['0L', '0R', '1L', '1R', '2L', '2R', '3L', '3R', '4L', '4R', '5L', '5R']
model = load_model(MODEL_PATH)
print(model.predict(image))
prediction = class_names[np.argmax(model.predict(image))]
print("Prediction Completed and output is {}".format(prediction))


#Print prediction on image
#label = "Mask" if mask > withoutMask else "No Mask"
#probability = mask*100 if mask > withoutMask else withoutMask*100
#color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
#cv2.putText(copy, "{} {:0.2f}".format(label, probability), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
# show the output image
#cv2.imshow("Output", copy)
#cv2.waitKey(0)