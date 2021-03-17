from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from TrainingMonitor import TrainingMonitor
import matplotlib.pyplot as plt




#Global declarations
BATCH_SIZE=4
EPOCHS=20
INIT_LR=1e-4





# Loading images.
# Ref: https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
print("Loading Images using ImageDataGenerator...")
traindatagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.15, horizontal_flip=True, fill_mode="nearest", rescale=1./255)

train_generator = traindatagen.flow_from_directory(directory="./dataset/train/", target_size=(224, 224), batch_size=BATCH_SIZE,
                                              class_mode="categorical")

#No data augmentation on test or validation images
testdatagen = ImageDataGenerator(rescale=1./255)
test_generator = testdatagen.flow_from_directory(directory="./dataset/test/", target_size=(224, 224), batch_size=BATCH_SIZE,
                                             class_mode="categorical")
#ImageDataGenerator does hot encoding on class labels based on class_mode parameter

# confirm the iterator works
batch_train_x, batch_train_y = train_generator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batch_train_x.shape, batch_train_x.min(), batch_train_x.max()))
#batchX.min()--min pixel intensity for this batch(0 in our case)
#batchX.max()--min pixel intensity for this batch(1 in our case)

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






#Preparing Model by finetuning
# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False







#Compile Model(this will also warm up the Head FC)
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
callbacks=[TrainingMonitor("perEpochGraph.png")]
#model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)




#Train the model
print("Training head...")
H = model.fit(train_generator, steps_per_epoch=len(train_generator) // BATCH_SIZE,
              validation_data=test_generator, validation_steps=len(test_generator) // BATCH_SIZE, epochs=EPOCHS,
              callbacks=callbacks)
print("Training completed!!")





# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_model.h5", save_format="h5")