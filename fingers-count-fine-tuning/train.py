from imutils import paths
import cv2
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.vis_utils import plot_model
from TrainingMonitor import TrainingMonitor
import imutils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from makeImageDataGenCompatibleDir import ImgGenCompatibleDir
import matplotlib.pyplot as plt



#Global declarations
BATCH_SIZE=1
EPOCHS=30
INIT_LR=1e-5





#Making ImgDataGen compatible directory
ImgGenCompatibleDir("./dataset/train/", "./dataset/test/", "./dataset/train_custom", "./dataset/test_custom").make()






print("Loading Images using ImageDataGenerator...")
traindatagen = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                                  shear_range=0.15, fill_mode="nearest", rescale=1./255)

train_generator = traindatagen.flow_from_directory(directory="./dataset/train_custom", target_size=(224, 224), batch_size=BATCH_SIZE,
                                              class_mode="categorical")
print(train_generator.class_indices)

#No data augmentation on test or validation images
testdatagen = ImageDataGenerator(rescale=1./255)
test_generator = testdatagen.flow_from_directory(directory="./dataset/test_custom", target_size=(224, 224), batch_size=BATCH_SIZE,
                                             class_mode="categorical")
#ImageDataGenerator does hot encoding on class labels based on class_mode parameter

# confirm the iterator works
batch_train_x, batch_train_y = train_generator.next()
#print('Batch shape=%s, min=%.3f, max=%.3f' % (batch_train_x.shape, batch_train_x.min(), batch_train_x.max()))
#batch_train_x.min()        #min pixel intensity for this batch(0 in our case)
#batch_train_x.max()        #min pixel intensity for this batch(1 in our case)

#How to iterate and shio images
#print(batch_train_y)
#for i in range(0, len(batch_train_x)):
#    image = batch_train_x[i]
#    label = batch_train_y[i]
#    print(label)
#    print(label.shape)
#    plt.imshow(image)
#    plt.show()
print("Loading of images completed!")







#always visualize the model before fine-tuning
#baseModel = ResNet50()
#plot_model(baseModel, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#exit(0)

#Preparing Model by finetuning
# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(12, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

#model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



#Compile Model(this will also warm up the Head FC)
print("compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])


#callbacks
filepath = "model-checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
modelCheckPoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks = [TrainingMonitor("./perEpochGraph.png"), modelCheckPoint]




#Train the model
print("Training head...")
H = model.fit(train_generator, steps_per_epoch=len(train_generator) // BATCH_SIZE,
              validation_data=test_generator, validation_steps=len(test_generator) // BATCH_SIZE, epochs=EPOCHS,
              callbacks=callbacks)
print("Training completed!!")



# serialize the model to disk
print("saving mask detector model...")
model.save("mask_model.h5", save_format="h5")