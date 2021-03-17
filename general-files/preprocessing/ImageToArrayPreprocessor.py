from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        #store the initialised value: if dataFormat=None Keras will use keras.json to figure out
        self.dataFormat=dataFormat

    def preprocess(self, image):
        #apply Keras utitlity function to rearrange dimensions of image
        print('Applying img_to_array utility of Keras to image')
        return img_to_array(image, data_format=self.dataFormat)