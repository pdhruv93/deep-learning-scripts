#Saves the model to the .hdf5 after every x epochs
#This script is not doing anything related to loading of model from a particular epoch#

from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):

    def __init__(self, outputPath, every=5, startAtEpoch=0):
        super(Callback, self).__init__()    #standard code: call the parent constructor
        self.outputPath = outputPath        #output path for the model
        self.every = every                  #save model to file after this epoch
        self.currentEpochCount = startAtEpoch    #resume at this epoch


    #overriding parent class method
    def on_epoch_end(self, epoch, logs={}):
        #you cannot rely on "epoch", as even if you have startAtEpoch=25, "epoch" will start from 0
        #when you serialize a model to disk, the last epoch number is never serialized
        #so at the time of resuming, you need to handle the last epoch manually by passing startAtEpoch parameter

        #check if the model should be serialized to disk
        if (self.currentEpochCount + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath, "epoch_{0}.hdf5".format(self.currentEpochCount + 1)])
            print("Checkpointing model for epoch:{0}".format(self.currentEpochCount + 1))
            self.model.save(p, overwrite=True)
            #How can we use self.model, its not there in callback function parameters
            #Check here:https://www.tensorflow.org/guide/keras/custom_callback
            #In addition to receiving log information when one of their methods is called,
            #callbacks have access to the model associated with the current round of training/evaluation/inference

        # increment the epoch counter
        self.currentEpochCount += 1