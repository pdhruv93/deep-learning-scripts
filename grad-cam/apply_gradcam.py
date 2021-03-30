# import the necessary packages
from gradcam import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.keras.models import load_model


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--model", required=True, help="model to be used")
args = vars(ap.parse_args())




# load the original image from disk and apply same transformations that you applied during training
orig = cv2.imread(args["image"])

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (128, 128))
image = cv2.GaussianBlur(image, (7,7), 3)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
ret, image = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
image = img_to_array(image)
image = image * 1./255
image = np.expand_dims(image, axis=0)




print("[INFO] loading model...")
model = load_model(args["model"])
class_names = ['L0', 'R0', 'L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'L4', 'R4', 'L5', 'R5']  #change
# use the network to make predictions on the input image and find
# the class label index with the largest corresponding probability
preds = model.predict(image)
print(preds)
i = np.argmax(preds[0])         #index which has highest probability
# decode the ImageNet predictions to obtain the human-readable label
label = class_names[np.argmax(preds)]
label = "{}: {:.2f}%".format(label, preds[0][i] * 100)
print("[INFO] {}".format(label))



# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)
# resize the resulting heatmap to the original input image dimensions and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)



# draw the predicted label on the output image
#cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
#cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
# display the original image and resulting heatmap and output image to our screen
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)