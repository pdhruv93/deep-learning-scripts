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
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.utils.vis_utils import plot_model
from TrainingMonitor import TrainingMonitor
import imutils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from makeImageDataGenCompatibleDir import ImgGenCompatibleDir
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D



#Global declarations
BATCH_SIZE=1
EPOCHS=30
INIT_LR=1e-5





#Making ImgDataGen compatible directory
ImgGenCompatibleDir("./dataset/train/", "./dataset/test/", "./dataset/train_custom", "./dataset/test_custom").make()






print("Loading Images using ImageDataGenerator...")
traindatagen = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                                  shear_range=0.15, fill_mode="nearest", rescale=1./255)

train_generator = traindatagen.flow_from_directory(directory="./dataset/train_custom", target_size=(96, 96), batch_size=BATCH_SIZE,
                                              class_mode="categorical")
print(train_generator.class_indices)

#No data augmentation on test or validation images
testdatagen = ImageDataGenerator(rescale=1./255)
test_generator = testdatagen.flow_from_directory(directory="./dataset/test_custom", target_size=(96, 96), batch_size=BATCH_SIZE,
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




#Preparing a CNN model from scratch : Ref from book1 pg232
#check here more about padding: https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t#:~:text=With%20%22SAME%22%20padding%2C%20if,only%20uses%20valid%20input%20data.
#Batch Normalization : pg232

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(96, 96, 3)))   #padding that make i/p and o/p same size
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3)))   #no padding
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), padding='same'))   #padding that make i/p and o/p same size
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3)))   #no padding
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))

#Final Layer
model.add(Dense(12))
model.add(Activation("softmax"))

model.summary()











#Compile Model
print("compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])


#callbacks
filepath = "./checkpoints/model-checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5"
modelCheckPoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks = [TrainingMonitor("./perEpochGraph.png"), modelCheckPoint]




#Train the model
print("Training head...")
H = model.fit(train_generator, steps_per_epoch=len(train_generator) // BATCH_SIZE,
              validation_data=test_generator, validation_steps=len(test_generator) // BATCH_SIZE, epochs=EPOCHS,
              callbacks=callbacks)
print("Training completed!!")



# serialize the model to disk
print("saving mask detector model...")
model.save("./finger_count_model.h5", save_format="h5")