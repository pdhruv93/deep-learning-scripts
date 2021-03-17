from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import model_from_json

#Global Declarations
MODEL_PATH = "model_seq_weights.h5"
IMAGE_PATH = "./to_predict_images/6.PNG"


#Load Image
image = cv2.imread(IMAGE_PATH)
copy = image.copy()
copy = cv2.resize(copy, (1000, 1000)) #this is just done bcoz the image was too big in the final output window and there was no minimize option
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (50, 50))
image = img_to_array(image)
image = image * 1./255
image = np.expand_dims(image, axis=0)
#print(image.shape)


#load model from weights
# load json and create model
json_file = open('model_seq_structure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(MODEL_PATH)
print("Loaded model from disk")

# evaluate loaded model on test data
# Define X_test & Y_test data first
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#Load model and Predict
class_names = ['L0', 'R0', 'L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'L4', 'R4', 'L5', 'R5']
#model = load_model(MODEL_PATH)
print(loaded_model.predict(image))
prediction = class_names[np.argmax(loaded_model.predict(image))]
print("Prediction Completed and output is {}".format(prediction))


#Print prediction on image
#label = "Mask" if mask > withoutMask else "No Mask"
#probability = mask*100 if mask > withoutMask else withoutMask*100
#color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
#cv2.putText(copy, "{} {:0.2f}".format(label, probability), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
# show the output image
cv2.imshow("Output", copy)
cv2.waitKey(0)