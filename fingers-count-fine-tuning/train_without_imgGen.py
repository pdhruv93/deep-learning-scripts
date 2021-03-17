from imutils import paths
import cv2
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from TrainingMonitor import TrainingMonitor
import imutils




#Global declarations
from sklearn.preprocessing import LabelEncoder

BATCH_SIZE=32
EPOCHS=20
INIT_LR=1e-4




#TRAIN DATA AND LABELS
train_data = []
train_labels = [] #0L, 1L, 2L...5L,0R, 1R....5R(Total 12 classes)
# Loading images.
print("Loading train Images and Labels")
for imagePath in sorted(list(paths.list_images("./dataset/train/"))):
    #load the train image, preprocess it and store it in the data list
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=224, height=224)  #Resnet50 needs 224 size for image
    #image.shape-->(224, 224, 3) ...use cv2.cvtColor to convert to Grayscale. But we are using Resnet50 so 3 channels are OK.
    image = img_to_array(image)
    train_data.append(image)

    #extract train class labels from image path
    label = os.path.splitext(imagePath)[0][-2:]
    train_labels.append(label)
print("Loading of train images completed!")





#scaling intensities between [0,1] and applying one-hot encoding to train labels
print("Adjusting train image intensities to [0,1]")
train_data = np.array(train_data, dtype="float") / 255.0
train_labels = np.array(train_labels)

print("One hot encoding train labels")
le = LabelEncoder().fit(train_labels)
train_labels = np_utils.to_categorical(le.transform(train_labels), 12)  #since we have 12 classes
#print(train_labels)









#TEST DATA AND LABELS
test_data = []
test_labels = [] #0L, 1L, 2L...5L,0R, 1R....5R(Total 12 classes)
# Loading images.
print("Loading test Images and Labels")
for imagePath in sorted(list(paths.list_images("./dataset/test/"))):
    #load the test image, preprocess it and store it in the data list
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=224, height=224)
    image = img_to_array(image)
    test_data.append(image)

    #extract test class labels from image path
    label = os.path.splitext(imagePath)[0][-2:]
    test_labels.append(label)
print("Loading of test images completed!")





#scaling intensities between [0,1] and applying one-hot encoding to test labels
print("Adjusting test image intensities to [0,1]")
test_data = np.array(test_data, dtype="float") / 255.0
test_labels = np.array(test_labels)

print("One hot encoding test labels")
le = LabelEncoder().fit(test_labels)
test_labels = np_utils.to_categorical(le.transform(test_labels), 12)  #since we have 12 classes
#print(test_labels)










#Handling class imbalance for training images(skew)
#print("Handling class Imbalance") -- No need in this dataset, already equal 1500 entries for each class
#classTotals = labels.sum(axis=0)
#classWeight = classTotals.max() / classTotals
#print(classTotals)
#print(classWeight)





#always visualize the model before fine-tuning
#baseModel = ResNet50()
#plot_model(baseModel, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#exit(0)


#Preparing Model by finetuning
# load the ResNet50 network, ensuring the head FC layer sets are left off
baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))


# construct the head of the model that will be placed on top of the the base model
#Ref : https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(12, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

#model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



#Compile Model(this will also warm up the Head FC)
print("compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
callbacks = [TrainingMonitor("perEpochGraph.png")]




#Train the model
print("Training head...")
H = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), steps_per_epoch=len(train_data) // BATCH_SIZE,
              epochs=EPOCHS, callbacks=callbacks)
print("Training completed!!")



# serialize the model to disk
print("saving mask detector model...")
model.save("mask_model.h5", save_format="h5")