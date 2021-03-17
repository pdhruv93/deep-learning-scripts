#we’ll create a TrainingMonitor callback that will be called at the
#end of every epoch when training a network with Keras. This monitor will serialize the loss and
#accuracy for both the training and validation set to disk, followed by constructing a plot of the data.
#Applying this callback during training will enable us to babysit the training process and spot
#overfitting early, allowing us to abort the experiment and continue trying to tune our parameters.

from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, graphPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__() #standard code when you inherit some class
        self.graphPath = graphPath  #path to output graph used to visualise loss and accuracy
        self.jsonPath = jsonPath #optional param to serialize loss and accuracy values to JSON file
        self.startAt = startAt #starting epoch where training is resumed in cntrl+C method

    #overriding method from BaseLogger class: this will be automatically called when training starts
    def on_train_begin(self, logs=None):
        #initilize a History dictionary. Holds following values: train_loss, train_acc, val_loss, val_acc
        #H={"train_loss": [80, 70...for all epochs], "train_acc": [10, 20...for all epochs], "val_loss"... }
        self.H = {}

        #if history exists in jsonPath file, load that history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt > 0:
                    #loop over entries in History dictionary and trim any entries past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]


    # overriding method from BaseLogger class: this will be automatically called when 1 epoch of training ends
    # Keras will automatically supply parameters to this function
    def on_epoch_end(self, epoch, logs=None):
        #logs variable holds acc , loss values for the current epoch
        #logs={"train_loss": 40, "train_acc": 80, "val_loss": 50, "val_acc": 90}
        for (k,v) in logs.items():  #k will be "train_loss", "train_loss", "val_loss", "val_acc"
            l=self.H.get(k, [])     #get existing value stored in H dictionary for this key
            l.append(v)             #append new value to the H dictionary
            self.H[k] =l            #update the dictionary with appended value

        if self.jsonPath is not None:
            #write new dictionary to jsonPath file
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        if len(self.H["loss"]) > 1:
            #if atleast 2 epochs have completed, then plot the loss and accuracy graph
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["acc"], label="train_acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Loss and Accuracy Graph for epoch {0}".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss or Accuracy")
            plt.legend()

            plt.savefig(self.graphPath)
            plt.close()