from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img


#Global Declarations
MODEL_PATH = "./mask_model.h5"
IMAGE_PATH = "./to_predict_images/4.jpg"


#Load Image
image = cv2.imread(IMAGE_PATH)
copy = image.copy()
copy = cv2.resize(copy, (1000, 1000)) #this is just done bcoz the image was too big in the final output window and there was no minimize option
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = img_to_array(image)
image = image * 1./255
image = np.expand_dims(image, axis=0)
#print(image.shape)



#Load model and Predict
model = load_model(MODEL_PATH)
(mask, withoutMask) = model.predict(image)[0]
print("Prediction Completed with probabilities: Mask:{:0.2f}, Without Mask:{:0.2f}".format(mask, withoutMask))


#Print prediction on image
label = "Mask" if mask > withoutMask else "No Mask"
probability = mask*100 if mask > withoutMask else withoutMask*100
color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
cv2.putText(copy, "{} {:0.2f}".format(label, probability), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
# show the output image
cv2.imshow("Output", copy)
cv2.waitKey(0)